import dgl
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SumPooling, GlobalAttentionPooling

# TODO ノード分類で試してみる
class GCNClassifier(nn.Module):
    # 馬の最大頭数は18頭なので分類は18classとする
    def __init__(self, in_feat=67, hidden_feat=2048, n_classifier=18, pool_type='mean'):
        super(GCNClassifier, self).__init__()
        self.conv1 = dglnn.SAGEConv(in_feat, hidden_feat, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(hidden_feat, hidden_feat, aggregator_type='mean')
        self.conv3 = dglnn.SAGEConv(hidden_feat, hidden_feat, aggregator_type='mean')
        self.fc1 = nn.Linear(hidden_feat, hidden_feat)
        self.fc2 = nn.Linear(hidden_feat, 1024)
        self.fc3 = nn.Linear(1024, n_classifier)

        if pool_type == 'mean':
            self.pool = AvgPooling()
        elif pool_type == 'max':
            self.pool = MaxPooling()
        elif pool_type == 'sum':
            self.pool = SumPooling()
        elif pool_type == 'attention':
            self.gate_nn = nn.Linear(hidden_feat, 1)
            self.pool = GlobalAttentionPooling(self.gate_nn)
        else:
            raise NotImplementedError

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.5)
        self.batch_norm1 = nn.BatchNorm1d(hidden_feat)
        self.batch_norm2 = nn.BatchNorm1d(hidden_feat)
        self.batch_norm3 = nn.BatchNorm1d(hidden_feat)

    def forward(self, g, h):
        # Apply graph convolution and activation.
        h = F.leaky_relu(self.batch_norm1(self.conv1(g, h)))
        h = F.leaky_relu(self.batch_norm2(self.conv2(g, h)))
        h = F.leaky_relu(self.batch_norm3(self.conv3(g, h)))
        hg = self.pool(g, h)
        hg = self.dropout1(hg)
        x = F.leaky_relu(self.fc1(hg))
        x = self.dropout2(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
        # with g.local_scope():
            # g.ndata['h'] = h
            # # Calculate graph representation by average readout.
            # hg = dgl.mean_nodes(g, 'h')
            # # 上の3行はAvePooling()で置き換えることができる
            # x = F.relu(self.fc1(hg))
            # x = self.dropout(x)
            # x = self.fc2(x)
            # return x

class NodeClassifier(nn.Module):
    def __init__(self, in_feats=67, hidden_feats=256, out_feats=2):
        super(NodeClassifier, self).__init__()
        self.conv1 = dglnn.GraphConv(
            in_feats=in_feats, out_feats=hidden_feats)
        self.conv2 = dglnn.GraphConv(
            in_feats=hidden_feats, out_feats=out_feats)

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h

class Classifier(nn.Module):
    def __init__(self, in_feats=48, hidden_feats=256, out_feats=2):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_feats, hidden_feats)
        self.fc2 = nn.Linear(hidden_feats, hidden_feats)
        self.fc3 = nn.Linear(hidden_feats, hidden_feats)
        self.fc4 = nn.Linear(hidden_feats, out_feats)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)

        return x
