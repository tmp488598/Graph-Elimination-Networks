import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import global_mean_pool, global_add_pool
from models import GENsConv
from torch_geometric.nn import GATConv,GCNConv


class GNN(torch.nn.Module):

    def __init__(self, dataset=None, hidden=128, num_conv_layers=3,
                 num_fc_layers=2, gfn=False, collapse=False, batch_norm=True, residual=False,cat =False,
                global_pool="sum", dropout=0, edge_norm=True):
        super(GNN, self).__init__()

        if "sum" in global_pool:
            self.global_pool = global_add_pool
        else:
            self.global_pool = global_mean_pool

        self.dropout = dropout
        self.batch_norm = batch_norm
        self.cat = cat
        
        hidden_in = dataset.num_features
        
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.convs.append(GENsConv(hidden_in, hidden, edge_dim=9))
            self.bns_conv.append(BatchNorm1d(hidden_in))
            hidden_in=hidden

        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))

        self.lin_class = Linear(hidden, int(dataset.num_classes))


    def masked_softmax(self, src, mask, dim=-1):
        out = src.masked_fill(~mask, float('-inf'))
        out = torch.softmax(out, dim=dim)
        out = out.masked_fill(~mask, 0)
        return out

    def forward(self, data, index_ji=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = torch.zeros(len(edge_index[0]), 9).to(edge_index.device)
        edge_attr[:, 7] = 1

        xs,next = [x],None
        for conv, batch_norm in zip(self.convs, self.bns_conv):
            x = batch_norm(xs[-1]) if self.batch_norm else xs[-1]
            x= conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)

        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]


        x = self.global_pool(x, batch)

        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x_

        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__



