import re

import pandas as pd

def horse_process(is_local=False):
    df = pd.read_csv('horse_data.csv')

    # データが存在しないものを削除
    df = df.drop(columns=['映像','ﾀｲﾑ指数','厩舎ｺﾒﾝﾄ','備考'])

    # 不必要なデータを削除
    df = df.drop(columns=['勝ち馬(2着馬)'])

    if is_local:
        pass
    else:
        # 地方競馬場はラウンド表記がないため数字の正規表現で地方競馬場の結果は省ける
        df = df[df['開催'].str.match('[0-9]+')]

    # ==============ここからNANデータの処理==============

    # ==============ここから距離適性の導出==============

    # ==============ここから脚質の導出==============


    df.to_csv('preprocessed_horse_data.csv', index=False)

def race_process():
    df = pd.read_csv('race_data.csv')

def main():
    pass

if __name__=='__main__':
    horse_process()