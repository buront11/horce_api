import dgl
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F

# TODO ノード分類で試してみる
class GCNClassifier(nn.Module):
    # 馬の最大頭数は18頭なので分類は18classとする
    def __init__(self, in_feat=67, hidden_feat=256, n_classifier=18):
        super(GCNClassifier, self).__init__()
        self.conv1 = dglnn.SAGEConv(in_feat, hidden_feat, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(hidden_feat, hidden_feat, aggregator_type='mean')
        self.fc1 = nn.Linear(hidden_feat, n_classifier)

    def forward(self, g, h):
        # Apply graph convolution and activation.
        h = F.leaky_relu(self.conv1(g, h))
        h = F.leaky_relu(self.conv2(g, h))
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h')
            x = self.fc1(hg)
            return x

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
