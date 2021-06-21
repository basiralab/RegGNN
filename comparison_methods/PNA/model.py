'''Files required for PNA implementation.

Code taken from https://github.com/lukecavabarrett/pna/
'''
from . import pna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree


class PNANet(nn.Module):
    '''
    Basic PNA model where the GCN layers in RegGNN architecture is replaced
    with PNA layers with variable aggregation and scaling methods.
    '''

    def __init__(self, nfeat, nhid, nclass, dropout, aggrs, scalers, deg):
        super(PNANet, self).__init__()
        aggregators = aggrs
        scalers = scalers

        self.dropout = dropout
        self.gc1 = pna.PNAConv(nfeat, nhid, aggregators=aggregators,
                               scalers=scalers, deg=deg, post_layers=1)
        self.gc2 = pna.PNAConv(nhid, nclass, aggregators=aggregators,
                               scalers=scalers, deg=deg, post_layers=1)
        self.dropout = dropout
        self.LinearLayer = nn.Linear(nfeat, 1)

    def forward(self, x, edge_index, edge_weights, batch):
        x = F.relu(self.gc1(x, edge_index, None))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index, None)

        x = self.LinearLayer(torch.transpose(x, 0, 1))
        return torch.transpose(x, 0, 1)

    def loss(self, pred, score):
        return F.mse_loss(pred, score)


def compute_deg(train_loader):
    deg = torch.zeros(117, dtype=torch.long)
    for data in train_loader:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg
