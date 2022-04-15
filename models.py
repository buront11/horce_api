import dgl
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch import GATv2Conv
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.glob import (AvgPooling, GlobalAttentionPooling,
                                 MaxPooling, SumPooling)

from config import Parameters

# TODO ノード分類で試してみる
class GCNClassifier(nn.Module):
    # 馬の最大頭数は18頭なので分類は18classとする
    def __init__(self, in_feat=63, hidden_feat=2048, n_classifier=18, gnn_dropout=0.3, affine_dropout=0.5,
        aggregator_type='mean',pool_type='mean'):
        super(GCNClassifier, self).__init__()
        self.conv1 = dglnn.SAGEConv(in_feat, hidden_feat, aggregator_type=aggregator_type)
        self.conv2 = dglnn.SAGEConv(hidden_feat, hidden_feat, aggregator_type=aggregator_type)
        self.conv3 = dglnn.SAGEConv(hidden_feat, hidden_feat, aggregator_type=aggregator_type)
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

        self.dropout1 = nn.Dropout(p=gnn_dropout)
        self.dropout2 = nn.Dropout(p=affine_dropout)
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

class GATClassifier(nn.Module):
    # 馬の最大頭数は18頭なので分類は18classとする
    def __init__(self, num_heads, in_feat=63, hidden_feat=1024, n_classifier=18, gnn_dropout=0.3, affine_dropout=0.5,
        feat_drop=0.1,attn_drop=0.1, pool_type='mean'):
        super(GATClassifier, self).__init__()
        self.conv1 = dglnn.GATv2Conv(in_feat, hidden_feat, num_heads=num_heads[0], feat_drop=feat_drop, attn_drop=attn_drop)
        self.conv2 = dglnn.GATv2Conv(hidden_feat*num_heads[0], hidden_feat, num_heads=num_heads[1], feat_drop=feat_drop, attn_drop=attn_drop)
        self.conv3 = dglnn.GATv2Conv(hidden_feat*num_heads[1], hidden_feat, num_heads=num_heads[2], feat_drop=feat_drop, attn_drop=attn_drop)
        self.fc1 = nn.Linear(hidden_feat*num_heads[2], hidden_feat)
        self.fc2 = nn.Linear(hidden_feat, 1024)
        self.fc3 = nn.Linear(1024, n_classifier)

        if pool_type == 'mean':
            self.pool = AvgPooling()
        elif pool_type == 'max':
            self.pool = MaxPooling()
        elif pool_type == 'sum':
            self.pool = SumPooling()
        elif pool_type == 'attention':
            self.gate_nn = nn.Linear(hidden_feat*num_heads[2], 1)
            self.pool = GlobalAttentionPooling(self.gate_nn)
        else:
            raise NotImplementedError

        self.dropout1 = nn.Dropout(p=gnn_dropout)
        self.dropout2 = nn.Dropout(p=affine_dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_feat*num_heads[0])
        self.batch_norm2 = nn.BatchNorm1d(hidden_feat*num_heads[1])
        self.batch_norm3 = nn.BatchNorm1d(hidden_feat*num_heads[2])

    def forward(self, g, h):
        # Apply graph convolution and activation.
        h = F.leaky_relu(self.batch_norm1(self.conv1(g, h).flatten(1)))
        h = F.leaky_relu(self.batch_norm2(self.conv2(g, h).flatten(1)))
        h = F.leaky_relu(self.batch_norm3(self.conv3(g, h).flatten(1)))
        hg = self.pool(g, h)
        hg = self.dropout1(hg)
        x = F.leaky_relu(self.fc1(hg))
        x = self.dropout2(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

class GATv2(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GATv2, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        if num_layers > 1:
        # input projection (no residual)
            self.gat_layers.append(GATv2Conv(
                in_dim, num_hidden, heads[0],
                feat_drop, attn_drop, negative_slope, False, self.activation))
            # hidden layers
            for l in range(1, num_layers-1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(GATv2Conv(
                    num_hidden * heads[l-1], num_hidden, heads[l],
                    feat_drop, attn_drop, negative_slope, residual, self.activation))
            # output projection
            self.gat_layers.append(GATv2Conv(
                num_hidden * heads[-2], num_classes, heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None))
        else:
            self.gat_layers.append(GATv2Conv(
                in_dim, num_classes, heads[0],
                feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, g, h):
        for l in range(self.num_layers-1):
            h = self.gat_layers[l](g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](g, h).mean(1)
        return logits

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

if __name__ == '__main__':
    params = Parameters()
    