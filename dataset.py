import networkx as nx
import pandas as pd

import torch
from torch.utils.data import Dataset

import dgl

from preprocess import preprocess

class GCNDataset(Dataset):
    def __init__(self, nn_type='graph') -> None:
        super(GCNDataset, self).__init__()
        df = pd.read_csv('dataset.csv')

        self.datas = []
        self.labels = []

        if nn_type == 'graph':
        # グラフに変換
        # TODO いつかレース情報を別のノードとする
            for race_id in df['race_id'].unique():
                race_df = df[df['race_id'] == race_id]

                race_df = race_df.sort_values('horse_num').reset_index(drop=True)

                graph = nx.cycle_graph(len(race_df))

                # TODO horse_numberをdropするかいなかを試す
                for index, row in enumerate(race_df.drop(['ranking', 'horse_num'], axis=1).values.tolist()):
                    graph.nodes[index]['feat'] = row

                for index, row in enumerate(race_df['ranking'].values.tolist()):
                    if row == 1:
                        graph.nodes[index]['label'] = 1
                    else:
                        graph.nodes[index]['label'] = 0

                dgl_graph = dgl.from_networkx(graph, node_attrs=['feat','label'], device='cpu')

                label = race_df[race_df['ranking'] == 1].index[0]

                self.datas.append(dgl_graph)
                # self.labels.append(label)
        else:
            for row in df.drop(columns=['ranking']).values.tolist():
                self.datas.append(row)

            for row in df['ranking'].values.tolist():
                if row == 1 or row == 2 or row == 3:
                    self.labels.append(1)
                else:
                    self.labels.append(0)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        out_data = torch.tensor(self.datas[index])
        out_label = torch.tensor(self.labels[index])

        return out_data, out_label

if __name__ == '__main__':
    GCNDataset(nn_type='nn')