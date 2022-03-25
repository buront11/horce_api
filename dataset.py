import torch
from torch.utils.data import Dataset

from preprocess import preprocess

class GCNDataset(Dataset):
    def __init__(self) -> None:
        super(GCNDataset, self).__init__()
        df = preprocess()

        # ここからグラフ化する
        # TODO いつかレース情報を別のノードとする
        for race_id in df['race_id'].unique():
            race_df = df[df['race_id'] == race_id]
            print(race_df)
            dd


if __name__ == '__main__':
    GCNDataset()