import re

import numpy as np
import pandas as pd

def ranking_category2num(ranking):
    """降格などを含む順位を数値へと変換するmethod

    Parameters
    ----------
    ranking : str
        順位の情報

    Returns
    -------
    int
        数値となった順位の情報
    """
    ranking = str(ranking)
    return int(re.search(r'[0-9]+', ranking).group())

def distance_aptitude(df):
    for horse_id in df['horse_id'].unique():
        horse_df = df[df['horse_id'] == horse_id]

        # データ量が2以下(出場レースが2以下)の場合は一律で1600mとする(データ量が少ないため出せないor出せても信頼性に欠けるため)
        if len(horse_df) <= 2:
            meter_apt = 1600
            df.loc[df['horse_id'] == horse_id, 'meter_apt'] = meter_apt
            continue

        distances = []
        weights = []

        for row in horse_df.itertuples():
            w = abs((int(row.starters_num) - int(row.ranking) + 1)/int(row.starters_num))

            distances.append(row.meter)
            weights.append(w)
        
        meter_apt = np.average(distances, weights=weights)

        df.loc[df['horse_id'] == horse_id, 'meter_apt'] = meter_apt

    return df

def running_style(df):
    
    for horse_id in df['horse_id'].unique():
        horse_df = df[df['horse_id'] == horse_id]

        race = []
        weight = []
        for row in horse_df.itertuples():
            passing_result = [int(x) for x in row.passing.split('-')]
            mean_val = sum(passing_result)/len(passing_result)
            w = abs((int(row.starters_num) - int(row.ranking) + 1)/int(row.starters_num))
            race.append(mean_val)
            weight.append(w)
        
        # 馬の情報が2レース以下の場合適当に6の値とする
        if len(horse_df) <= 2:
            df.loc[df['horse_id'] == horse_id, 'run_style'] = 6
        else:
            df.loc[df['horse_id'] == horse_id, 'run_style'] = np.average(race, weights=weight)
        
    return df

def total_winning(df):
    # 賞金の獲得総額を取得する
    for horse_id in df['horse_id'].unique():
        horse_df = df[df['horse_id'] == horse_id]

        total_winning = 0
        total_winnings = []
        for money in horse_df[::-1].loc[:,'winning']:
            total_winning += money
            total_winnings.append(total_winning)

        total_winnings = list(reversed(total_winnings))
        df.loc[df['horse_id'] == horse_id, 'total_winning'] = total_winnings

    return df

def split_sex_old(df):
    sexs = []
    olds = []
    for row in df['sex_old']:
        sexs.append(re.sub(r'[0-9]+', '', row))
        olds.append(re.search(r'[0-9]+', row).group())

    df['sex'] = sexs
    df['old'] = olds

    return df

def split_horse_weight(df):
    horse_weights = []
    delta_weights = []
    for row in df['horse_weight']:
        delta_weights.append(int(re.search(r'(?<=\().+?(?=\))', row).group()))
        horse_weights.append(int(re.sub(r'\(.+\)', '', row)))

    df['horse_weight'] = horse_weights
    df['delta_weight'] = delta_weights

    return df

def horse_process():
    df = pd.read_csv('horse_data.csv')

    # 列名を英語に変換
    df = df.rename(columns={'日付':'day','開催':'hold','天気':'weather', 'R':'round', 
    'レース名':'race_name', '映像':'movie', '頭数':'starters_num', '枠番':'flame', '馬番':'horse_num', 'オッズ':'odds',
    '人気':'popularity','着順':'ranking', '騎手':'jockey', '斤量':'weight', '距離':'meter', '馬場':'state',
    '馬場指数':'state_value', 'タイム':'time', '着差':'arrival_diff', 'ﾀｲﾑ指数':'time_value', '通過':'passing', 'ペース':'pace',
    '上り':'3f', '馬体重':'horse_weight', '厩舎ｺﾒﾝﾄ':'comment', '備考':'other', '勝ち馬(2着馬)':'second', '賞金':'winning'})

    # データが存在しないものを削除
    df = df.drop(['movie', 'comment', 'other', 'time_value', 'state_value'], axis=1)

    # 地方競馬場はラウンド表記がないため数字の正規表現で地方競馬場の結果は省ける
    # 地方競馬場のデータは削除する
    df = df[df['hold'].str.match(r'[0-9]+', na=False)]

    # レースの馬の頭数を4以上18以下に限定する
    df = df[(df['starters_num'] <= 18) & (df['starters_num'] >= 4)]

    # 取り消しや中止などの出走しなかったレースの情報を削除
    df = df[df['ranking'].str.match(r'[0-9]+', na=False)]

    # nan値は賞金を獲得できていないため0でfillする
    df = df.fillna({'winning':0})

    # レースの馬場(芝、ダート)はレース側で持ってこれるデータなので削除する
    df['meter'] = df['meter'].apply(lambda x:int(re.search(r'[0-9]+', x).group()))

    # 順位を数字に変換
    df['ranking'] = df['ranking'].apply(lambda x: ranking_category2num(x))

    # 距離適性の導出
    df = distance_aptitude(df)

    # 脚質の導出
    df = running_style(df)

    # 累計獲得賞金の導出
    df = total_winning(df)

    df.to_csv('preprocessed_horse_data.csv', index=False)

def race_process():
    df = pd.read_csv('race_data.csv')

    df = df.rename(columns={'着順':'ranking', '枠番':'flame', '馬番':'starters_num',\
            '性齢':'sex_old', '斤量':'weight', '騎手':'jockey', 'タイム':'time',\
            '着差':'arrival_diff', '単勝':'odds', '人気':'popularity','馬体重':'horse_weight', '調教師':'trainer'})

    # 馬のデータと結合する際に重複する部分を削除
    df = df.drop()

    # 出走しなかったレースを削除
    df = df[df['ranking'].str.match(r'[0-9]+', na=False)]

    # 順位を数字に変換
    df['ranking'] = df['ranking'].apply(lambda x: ranking_category2num(x))

    df = split_horse_weight(df)

    df = split_sex_old(df)

    df.to_csv('preprocessed_race_data.csv', index=False)

def main():
    pass

if __name__=='__main__':
    horse_process()