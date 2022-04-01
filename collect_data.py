from logging import raiseExceptions
import os
import re
import glob
import json
from tqdm import tqdm
import argparse

import time
from datetime import date, datetime, timedelta
from numpy import not_equal
import numpy as np

import pandas as pd

import requests
import cchardet
from bs4 import BeautifulSoup
from selenium import webdriver

def date_range(start, stop, step = timedelta(1)):
    current = start
    while current < stop:
        yield current
        current += step

def split_race_info(race_infos):
    # リストの一番目はレースの内容(距離などが入っている)
    # １文字目には芝かダートかの情報があるため除く
    wise = re.sub(r'[0-9]+m', '', race_infos[0])[1:]
    meter = re.search(r'[0-9]+m', race_infos[0]).group()[:-1]

    # リストの二つ目は天候の情報
    weather = re.sub(r'天候:', '', race_infos[1])

    # リストの三つ目は馬場の情報
    race_type = race_infos[2].split(':')[0]
    race_state = race_infos[2].split(':')[1]

    # リストの四つ目は発走情報
    start_time = re.sub(r'発走:', '', race_infos[3])

    return race_type, wise, meter, weather, race_state, start_time

class HorceDateCollecter():
    def __init__(self):
        # 前回の実行時刻を取得
        try:
            with open('last_updata_log.txt', 'r') as f:
                self.start_date = datetime.strptime(f.read(), '%Y-%m-%d').date()
        except FileNotFoundError:
            # 前回のデータがない場合は2008年1月1日からスクレイピング
            self.start_date = datetime(2012, 1, 1).date()
            
        # 前回のデータとの差分をとって最新のデータのみ持ってくる
        self.dt_today = datetime.utcnow().date()

        # try:
        #     self.race_df = pd.read_csv('race_data.csv')
        # except:
        self.race_df = pd.DataFrame(columns=["着順","枠番","馬番","馬名","性齢","斤量","騎手",\
                                            "タイム","着差","単勝","人気","馬体重","調教師",\
                                            "race_id","horse_id","jockey_id","trainer_id"])

        # try:
        #     self.horse_df = pd.read_csv('horse_data.csv')
        # except:
        self.horse_df = pd.DataFrame(columns=['日付','開催','天気','R','レース名','映像','頭数','枠番','馬番','オッズ','人気',
                                            '着順','騎手','斤量','距離','馬場','馬場指数','タイム','着差','ﾀｲﾑ指数','通過','ペース',
                                            '上り','馬体重','厩舎ｺﾒﾝﾄ','備考','勝ち馬(2着馬)','賞金','horse_id','race_id','jockey_id'])

    def get_race_data(self):
        for date in date_range(self.start_date, self.dt_today):
            # データをとった日付を記録
            with open('last_updata_log.txt', 'w') as f:
                f.write(str(date))

            date = re.sub('-', '', str(date))
            url = 'https://db.netkeiba.com/race/list/' + date
            res = requests.get(url)
            res.encoding = cchardet.detect(res.content)["encoding"]

            soup = BeautifulSoup(res.content, 'lxml')
            hold_race = soup.select_one('div.race_kaisai_info')
            if hold_race is None:
                continue

            print('get date {} now ...'.format(date))

            race_urls = hold_race.find_all('a', title=re.compile(r'.+'))
            for index in tqdm(range(len(race_urls))):
                race = race_urls[index].get('href')
                url = 'https://db.netkeiba.com' + race
                race_id = re.search(r'[0-9]+', race).group()

                res = requests.get(url)
                res.encoding = cchardet.detect(res.content)["encoding"]

                soup = BeautifulSoup(res.content, 'lxml')

                # 出走馬情報
                df_horses = pd.read_html(res.content)[0]

                df_horses['race_id'] = race_id

                # 前処理として調教師の名前の東西は削除する
                df_horses['調教師'] = df_horses['調教師'].apply(lambda x: re.sub(r'\[.\] ', '', x))

                # 馬、騎手、調教師、所有者のidを取得し、名前に対応した辞書を作成する
                info_ids = soup.select('td.txt_l a')
                info_dicts = {}
                for info in info_ids:
                    info_type = re.search(r'[a-z]+', info.get('href')).group()
                    name = info.get_text()
                    id = re.search(r'[0-9]+', info.get('href')).group()

                    # 馬主はtableで持ってこれないため現状はスキップする
                    if info_type == 'owner':
                        continue

                    if info_type not in info_dicts.keys():
                        info_dicts.update({info_type:{name:id}})
                    else:
                        info_dicts[info_type].update({name:id})

                # webページから持ってきたtableに代入する用のlistを作成
                for info_type, values in info_dicts.items():
                    if info_type == "horse":
                        column = "馬名"
                    elif info_type == "jockey":
                        column = "騎手"
                    elif info_type == "trainer":
                        column = "調教師"
                    # なぜか馬主がないためコメントアウト
                    # else:
                    #     column = "馬主"
                    add_list = []
                    for row in df_horses[column]:
                        # idと名前を照合し、リストに追加
                        add_list.append(values[row])
                    # dfにidを新しい列として追加する
                    df_horses[f'{info_type}_id'] = add_list

                # レース情報　レース名/馬場距離/天候/状態/発走時刻
                # 一旦なしで
                race_title = soup.select_one('div.data_intro h1').get_text()
                if re.search(r'G[1-3]', race_title):
                    race_grade = re.search(r'G[1-3]', race_title).group()
                else:
                    race_grade = 'other'
                # race_round = soup.select_one('div.data_intro dl.racedata.fc dt').get_text()
                race_info = soup.select_one('div.data_intro diary_snap_cut span').get_text()

                race_info = re.sub(u'\xa0', '', race_info)
                race_infos = re.sub(' ', '', race_info).split('/')
                
                race_type, wise, meter, weather, race_state, start_time = split_race_info(race_infos=race_infos)

                df_horses['race_grade'] = race_grade
                df_horses['race_type'] = race_type
                df_horses['wise'] = wise
                df_horses['meter'] = meter
                df_horses['weather'] = weather
                df_horses['race_state'] = race_state
                df_horses['start_time'] = start_time

                self.race_df = pd.concat([self.race_df, df_horses], ignore_index=True)

                time.sleep(1)
            
            self.race_df.to_csv('race_data.csv', index=False)

    def get_horse_date(self):
        horse_ids = self.race_df['horse_id'].unique()

        # 馬の情報の取得
        for index in tqdm(range(len(horse_ids))):
            print('get {} datas ...'.format(horse_ids[index]))
            
            url = 'https://db.netkeiba.com/horse/' + str(horse_ids[index])

            res = requests.get(url)
            res.encoding = cchardet.detect(res.content)["encoding"]
            soup = BeautifulSoup(res.content, 'lxml')

            df_horses = pd.read_html(res.content, match='レース名')[0]

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
                    for row in df_horses[column]:
                        # idと名前を照合し、リストに追加
                        if row in values.keys():
                            add_list.append(values[row])
                        else:
                            add_list.append(np.NaN)
                    # dfにidを新しい列として追加する
                    df_horses[f'{info_type}_id'] = add_list
                else:
                    # dfにidを新しい列として追加する
                    df_horses[f'{info_type}_id'] = values

            df_horses['horse_id'] = horse_ids[index]

            self.horse_df = pd.concat([self.horse_df, df_horses], ignore_index=True)

            time.sleep(1)

        self.horse_df.to_csv('horse_data.csv', index=False)

    def get_correct_jockey_name(self):
        new_jockey_ids = {} 

        for jockey, id in self.jockey_ids.items():
            print('get {} datas ...'.format(jockey))

            url = 'https://db.netkeiba.com/jockey/' + id

            res = requests.get(url)
            res.encoding = res.apparent_encoding
            soup = BeautifulSoup(res.content, 'html.parser')

            jockey_name = soup.select_one('div.db_head_name.fc h1').get_text()
            jockey_name = re.sub(r'\n', '', jockey_name)
            jockey_name = re.sub(r'\(.+\)', '', jockey_name)
            jockey_name = re.sub(r'\s', '', jockey_name)

            if jockey_name not in new_jockey_ids.keys():
                new_jockey_ids.update({jockey_name:id})

            time.sleep(1)

        self.jockey_ids = new_jockey_ids
        with open('./data/jockeys.json', 'w') as f:
            json.dump(self.jockey_ids, f, ensure_ascii=False, indent=2)

    def _save_ids(self):
        # 馬のidを保存
        with open('./data/horses.json', 'w') as f:
            json.dump(self.horse_ids, f, ensure_ascii=False, indent=2)
        # 騎手のidを保存
        with open('./data/jockeys.json', 'w') as f:
            json.dump(self.jockey_ids, f, ensure_ascii=False, indent=2)
        # 調教師のidを保存
        with open('./data/trainers.json', 'w') as f:
            json.dump(self.trainer_ids, f, ensure_ascii=False, indent=2)
        # 所有者のidを保存
        with open('./data/owners.json', 'w') as f:
            json.dump(self.owner_ids, f, ensure_ascii=False, indent=2)

    def _fix_garbled_dict(self, dict):
        """文字化けしているデータを削除する
            文字化けしているデータはなぜか重複データなので問答無用で消して問題ない

        Parameters
        ----------
        dict : 
            [消したい辞書]
        """
        hiragana = re.compile('[\u3041-\u309F]+')
        katakana = re.compile('[\u30A1-\u30FF]+')
        kanji = regex.compile(r'\p{Script=Han}+')

        del_keys = []

        for key in dict.keys():
            if katakana.search(key):
                continue
            elif kanji.search(key):
                continue
            elif hiragana.search(key):
                continue
            else:
                del_keys.append(key)

        for key in del_keys:
            del dict[key]

    def fix_garbled_char(self):
        self._fix_garbled_dict(self.horse_ids)
        self._fix_garbled_dict(self.jockey_ids)
        self._fix_garbled_dict(self.trainer_ids)
        self._fix_garbled_dict(self.owner_ids)

        self._save_ids()

def complement_horse_name():
    """現状馬のデータをスクレイピングしても名前がjsonに登録されない馬がいるので
        それを補完するプログラム
    """
    with open('./data/horses.json') as f:
        horse_db = json.load(f)
    horse_csvs = glob.glob('./data/horse_data/*')
    horse_ids = [os.path.splitext(os.path.basename(path))[0] for path in horse_csvs]
    katakana = re.compile('[\u30A1-\u30FF]+')
    
    for i in tqdm(range(len(horse_ids))):
        horse_name = [k for k, v in horse_db.items() if v == horse_ids[i]][0]
        if katakana.search(horse_name):
            continue
        del horse_db[horse_name]
        url = 'https://db.netkeiba.com/horse/' + str(horse_ids[i])
        res = requests.get(url)
        res.encoding = res.apparent_encoding
        soup = BeautifulSoup(res.content, 'html.parser')
        horse_name = soup.select_one('div.db_head_name.fc div.horse_title h1').get_text()
        horse_name = katakana.search(horse_name).group()
        print(horse_name)
        # 空白の削除
        horse_name = re.sub(r'\s', '', horse_name)
        time.sleep(1)
        if horse_name not in horse_db.keys():
            horse_db.update({horse_name:horse_ids[i]})

    with open('./data/horses.json', 'w') as f:
            json.dump(horse_db, f, ensure_ascii=False, indent=2)
        
def main(args):

    horce_collect = HorceDateCollecter()

    horce_collect.get_race_data()

    horce_collect.get_horse_date()

    # horce_collect.get_correct_jockey_name()

    # horce_collect.fix_garbled_char()

    # complement_horse_name()
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--http_proxy', default='')
    parser.add_argument('--https_proxy', default='')

    args = parser.parse_args()

    main(args)
