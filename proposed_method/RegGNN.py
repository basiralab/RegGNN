'''RegGNN regression model architecture.

torch_geometric needs to be installed.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.dense import DenseGCNConv


class RegGNN(nn.Module):
    '''Regression using a DenseGCNConv layer from pytorch geometric.

       Layers in this model are identical to GCNConv.
    '''

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(RegGNN, self).__init__()

        self.gc1 = DenseGCNConv(nfeat, nhid)
        self.gc2 = DenseGCNConv(nhid, nclass)
        self.dropout = dropout
        self.LinearLayer = nn.Linear(nfeat, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.gc1(x, edge_index))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        x = self.LinearLayer(torch.transpose(x, 2, 1))

        return torch.transpose(x, 2, 1)

    def loss(self, pred, score):
        return F.mse_loss(pred, score)
