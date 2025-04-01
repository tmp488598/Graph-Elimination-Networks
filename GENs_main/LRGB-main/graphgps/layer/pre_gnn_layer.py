import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GraphConv
from .models import GENsConv
from torch_geometric.graphgym.config import cfg
from .neuralwalker_layer import WalkEncoder

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=0, bn_bool = True):
        super(MLP, self).__init__()
        self.bn_bool = bn_bool

        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()


        for _ in range(num_layers - 1):
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            in_channels = hidden_channels
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):

        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x) if self.bn_bool else x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

class GNNPreMP(nn.Module):
    def __init__(self, dim_in, dim_inner, layers_pre_mp, gnn_type='walk', heads=1):

        super(GNNPreMP, self).__init__()
        self.layers_pre_mp = layers_pre_mp
        self.gnn_type = gnn_type
        self.dropout = nn.Dropout(p=cfg.gnn.dropout)
        self.relu = nn.ReLU()

        self.convs = nn.ModuleList()
        if self.gnn_type=='mlp':
            self.convs = MLP(dim_in, dim_inner, dim_inner, layers_pre_mp, cfg.gnn.dropout)
        elif self.gnn_type=='walk':
            self.convs = WalkEncoder(dim_in, 'transformer')
        else:
            for i in range(layers_pre_mp):
                in_dim = dim_in if i == 0 else dim_inner
                if self.gnn_type == 'GCN':
                    self.convs.append(GCNConv(in_dim, dim_inner))
                elif self.gnn_type == 'GAT':
                    self.convs.append(GATConv(in_dim, dim_inner // heads, heads=heads, concat=True))
                elif self.gnn_type == 'GRAPHSAGE':
                    self.convs.append(SAGEConv(in_dim, dim_inner))
                elif self.gnn_type == 'GRAPHCONV':
                    self.convs.append(GraphConv(in_dim, dim_inner))
                else:
                    raise ValueError(f"Unsupported gnn_type: {gnn_type}. Choose from 'GCN', 'GAT', 'GraphSAGE', 'GraphConv'.")

    def forward(self, batch):
        """
        前向传播方法。

        Args:
            batch (torch_geometric.data.Batch): 输入的图批次数据，包含节点特征和边索引等信息。

        Returns:
            torch_geometric.data.Batch: 更新后的图批次数据。
        """

        if self.gnn_type =='walk':
            batch.walk_pe = torch.cat(
                [batch.walk_node_id_encoding, batch.walk_node_adj_encoding], dim=-1
            )
            batch = self.convs(batch)


        elif self.gnn_type =='mlp':
            x = self.convs(batch.x)
            x = self.relu(x)
            x = self.dropout(x)
            batch.x = x

        else:
            x, edge_index = batch.x, batch.edge_index
            for conv in self.convs:
                x = conv(x, edge_index)
                x = self.relu(x)
                x = self.dropout(x)

            batch.x = x

        return batch
