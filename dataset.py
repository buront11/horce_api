import networkx as nx
import pandas as pd

import torch
from torch.utils.data import Dataset

import dgl

from preprocess import preprocess

class GCNDataset(Dataset):
    def __init__(self) -> None:
        super(GCNDataset, self).__init__()
        df = pd.read_csv('dataset.csv')

        self.datas = []
        self.labels = []

        # グラフに変換
        # TODO いつかレース情報を別のノードとする
        for race_id in df['race_id'].unique():
            race_df = df[df['race_id'] == race_id]

            race_df = race_df.sort_values('horse_num').reset_index(drop=True)

            graph = nx.complete_graph(len(race_df))

            # TODO horse_numberをdropするかいなかを試す
            for index, row in enumerate(race_df.drop(['ranking'], axis=1).values.tolist()):
                graph.nodes[index]['feat'] = row

            dgl_graph = dgl.from_networkx(graph, node_attrs=['feat'], device='cuda')

            label = race_df[race_df['ranking'] == 1].index

            self.datas.append(graph)
            self.labels.append(label)

            

if __name__ == '__main__':
    GCNDataset()