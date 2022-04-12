import dgl
import networkx as nx
import pandas as pd
import torch
from torch.utils.data import Dataset

from preprocess import preprocess


class GCNDataset(Dataset):
    def __init__(self, device, pred_rank=1,nn_type='graph', csv_path='dataset.csv') -> None:
        super(GCNDataset, self).__init__()
        df = pd.read_csv(csv_path)

        self.datas = []
        self.labels = []
        self.nn_type = nn_type

        if self.nn_type == 'graph' or self.nn_type == 'node':
        # グラフに変換
        # TODO いつかレース情報を別のノードとする
            for race_id in df['race_id'].unique():
                race_df = df[df['race_id'] == race_id]

                # レースの頭数が4頭以下となった場合はデータとして扱わない
                if len(race_df) <= 4:
                    continue

                race_df = race_df.sort_values('horse_num').reset_index(drop=True)

                graph = nx.complete_graph(len(race_df))

                # TODO horse_numberをdropするかいなかを試す
                for index, row in enumerate(race_df.drop(['ranking','horse_num','race_id','horse_id','jockey_id','trainer_id'], axis=1).values.tolist()):
                    graph.nodes[index]['feat'] = row

                if self.nn_type=='node':
                    for index, row in enumerate(race_df['ranking'].values.tolist()):
                        if row == 1 or row==2 or row==3:
                            graph.nodes[index]['label'] = 1
                        else:
                            graph.nodes[index]['label'] = 0

                    dgl_graph = dgl.from_networkx(graph, node_attrs=['feat','label'], device=device)
                else:
                    # 一位の情報が前処理によって消されたデータは削除する
                    try:
                        label = race_df[race_df['ranking'] == int(pred_rank)].index[0]
                        self.labels.append(label)
                    except:
                        continue

                    dgl_graph = dgl.from_networkx(graph, node_attrs=['feat'], device=device)

                self.datas.append(dgl_graph)

        else:
            # TODO 現状IDを特徴量として使うと学習がうまくいかなくなる問題がある
            for row in df.drop(columns=['ranking','race_id','horse_id','jockey_id','trainer_id']).values.tolist():
                self.datas.append(row)

            for row in df['ranking'].values.tolist():
                if row == 1 or row == 2 or row == 3:
                    self.labels.append(1)
                else:
                    self.labels.append(0)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        if self.nn_type=='node':
            out_data = torch.tensor(self.datas[index])

            return out_data
        else:
            out_data = self.datas[index]
            out_label = torch.tensor(self.labels[index])

            return out_data, out_label


if __name__ == '__main__':
    GCNDataset(nn_type='nn')
