import re
import time
import numpy as np
import pandas as pd
import requests
import cchardet
import networkx as nx
import dgl
from bs4 import BeautifulSoup

from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver

from preprocess import api_preprocess
from dataset import GCNDataset
from train import predict_race

def get_horse_data(horse_ids):
    horse_df = pd.DataFrame(columns=['日付','開催','天気','R','レース名','映像','頭数','枠番','馬番','オッズ','人気',
                                    '着順','騎手','斤量','距離','馬場','馬場指数','タイム','着差','ﾀｲﾑ指数','通過','ペース',
                                    '上り','馬体重','厩舎ｺﾒﾝﾄ','備考','勝ち馬(2着馬)','賞金','horse_id','race_id','jockey_id'])

    # 馬の情報の取得
    for horse_id in horse_ids:
        url = 'https://db.netkeiba.com/horse/' + str(horse_id)

        res = requests.get(url)
        res.encoding = cchardet.detect(res.content)["encoding"]
        soup = BeautifulSoup(res.content, 'lxml')

        race_df = pd.read_html(res.content, match='レース名')[0]

        # 馬、騎手、調教師、所有者のidを取得し、名前に対応した辞書を作成する
        info_ids = soup.select('table.db_h_race_results.nk_tb_common td a')
        info_dicts = {}
        for info in info_ids:
            info_type = re.search(r'[a-z]+', info.get('href')).group()
            # レースの場合リンクがいくつか存在するため欲しい情報のみに絞る
            if info_type == 'race':
                if re.search(r'/race/[0-9]+', info.get('href')):
                    pass
                else:
                    continue

            name = info.get_text()
            # 地方ジョッキーなどだと英語が入るためいくつか怪しいデータになる
            # 処理方法は後で考える
            id = re.search(r'[0-9]+', info.get('href')).group()

            # 逆にジョッキーにリンクがないものが存在する
            # 同じレース名がある場合に不具合が発生する
            if info_type == 'jockey':
                if info_type not in info_dicts.keys():
                    info_dicts.update({info_type:{name:id}})
                else:
                    info_dicts[info_type].update({name:id})
            else:
                if info_type not in info_dicts.keys():
                    info_dicts.update({info_type:[id]})
                else:
                    info_dicts[info_type].append(id)

        for info_type, values in info_dicts.items():
            if info_type == "race":
                column = "レース名"
            elif info_type == "jockey":
                column = "騎手"
            else:
                continue

            # ジョッキーにリンクがない人がいたので特殊な処理をする
            if info_type == 'jockey':
                add_list = []
                for row in race_df[column]:
                    # idと名前を照合し、リストに追加
                    if row in values.keys():
                        add_list.append(values[row])
                    else:
                        add_list.append(np.NaN)
                # dfにidを新しい列として追加する
                race_df[f'{info_type}_id'] = add_list
            else:
                # dfにidを新しい列として追加する
                race_df[f'{info_type}_id'] = values

        race_df['horse_id'] = horse_id

        horse_df = pd.concat([horse_df, race_df], ignore_index=True)

        time.sleep(1)

    return horse_df

def df2graph(df):
    df = df.sort_values('horse_num').reset_index(drop=True)

    graph = nx.complete_graph(len(df))

    # TODO horse_numberをdropするかいなかを試す
    for index, row in enumerate(df.drop(['horse_num','race_id','horse_id','jockey_id'], axis=1).values.tolist()):
        graph.nodes[index]['feat'] = row

    dgl_graph = dgl.from_networkx(graph, node_attrs=['feat'], device='cpu')

    return dgl_graph

def main(args):

    options = webdriver.ChromeOptions()

    options.add_argument('--headless')
    options.add_argument("--no-sandbox")

    chrome_service = webdriver.chrome.service.Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=chrome_service, options=options)

    url = 'https://race.netkeiba.com/race/shutuba.html?race_id=202209020611&rf=race_submenu'

    driver.get(url)

    race_df = pd.read_html(driver.page_source)[0]
    race_df.columns = race_df.columns.droplevel(0)

    driver.quit()

    res = requests.get(url)
    res.encoding = cchardet.detect(res.content)["encoding"]

    soup = BeautifulSoup(res.content, 'lxml')

    horse_ids = []
    info_ids = soup.select('td.HorseInfo a')
    jockey_ids = []
    jockey_id = soup.select('td.Jockey a')

    for info in info_ids:
        id = re.search(r'[0-9]+', info.get('href')).group()
        horse_ids.append(id)

    for info in jockey_id:
        id = re.search(r'[0-9]+', info.get('href')).group()
        jockey_ids.append(int(id))

    race_info = soup.select('div.RaceData01')
    race_infos = re.sub(r' ','',race_info[0].get_text()).split('/')

    race_type = race_infos[1][0]
    if race_type == 'ダ':
        race_type = 'ダート'
    wise = re.sub(r'(\(|\))', '', re.search(r'\(.+\)', race_infos[1]).group())
    meter = int(re.search(r'[0-9]+', race_infos[1]).group())
    weather = re.sub(r'(.+:|\n)', '', race_infos[2])
    race_state = re.sub(r'(.+:|\n)', '', race_infos[3])
    if race_state == '稍':
        race_state = '稍重'
    start_time = re.sub(r'(発走|\n)', '', race_infos[0])

    race_grade_info = soup.select('div.RaceName span.Icon_GradeType')
    if len(race_grade_info) == 0:
        race_grade = 'other'
    elif int(re.search(r'[0-9]', race_grade_info[0].get('class')[1]).group()) >= 4:
        race_grade = 'other'
    else:
        race_grade = 'G' + re.search(r'[0-9]', race_grade_info[0].get('class')[1]).group()

    horse_df = get_horse_data(horse_ids)

    race_df['horse_id'] = horse_ids
    race_df['jockey_id'] = jockey_ids
    race_df['race_grade'] = race_grade
    race_df['race_type'] = race_type
    race_df['wise'] = wise
    race_df['meter'] = meter
    race_df['weather'] = weather
    race_df['race_state'] = race_state
    race_df['start_time'] = start_time

    df = api_preprocess(race_df, horse_df)

    graph = df2graph(df)

    if args.model_weight == 'all':
        print('1着の予想確率')
        predict_race(graph, 'weight_rank_1')
        print('2着の予想確率')
        predict_race(graph, 'weight_rank_2')
        print('3着の予想確率')
        predict_race(graph, 'weight_rank_3')
    else:
        predict_race(graph, args.model_weight)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_weight', '-m', default='all')

    args = parser.parse_args()

    main(args)