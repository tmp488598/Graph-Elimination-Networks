import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_sparse import SparseTensor, fill_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from typing import Optional, Tuple
from torch import Tensor
from torch_sparse import remove_diag, matmul, mul
from torch_sparse import sum as sparsesum
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import softmax, degree
from torch_geometric.nn import DeepGCNLayer, GENConv


class DGMLP(nn.Module):
    '''
    The network implementation is from the paper "Evaluating
        Deep Graph Neural Networks"
    <https://arxiv.org/abs/2108.00955>
    '''

    def __init__(self, nfeat, nhid, dropout=0, num_hops=9, bn=True):
        super(DGMLP, self).__init__()
        self.bn = bn
        print(num_hops)
        self.num_layers = num_hops
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid))
        for i in range(self.num_layers - 1):
            self.convs.append(GCNConv(nhid, nhid))
            self.bns.append(torch.nn.BatchNorm1d(nhid))
        self.dropout = dropout
        self.lr_att = nn.Linear(nhid + nhid, 1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lr_att.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj):
        num_node = x.shape[0]
        x = self.convs[0](x, adj)
        x_input = x
        for i in range(1, self.num_layers):
            alpha = torch.sigmoid(self.lr_att(
                torch.cat([x, x_input], dim=1))).view(num_node, 1)
            x = (1 - alpha) * x + (alpha) * x_input
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.bns[i - 1](self.convs[i](x, adj)) if self.bn else self.convs[i](x, adj)
        return x


class GraphCON(nn.Module):
    '''
    The network implementation is from the paper "Graph-Coupled
        Oscillator Networks"
    <https://arxiv.org/abs/2202.02296>
    '''

    def __init__(self, hiddin, dt=1., alpha=0.2, gamma=1, num_layers = 6,dropout=0.0):
        super(GraphCON, self).__init__()
        self.dt = dt
        self.alpha = alpha
        self.gamma = gamma
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(GCNConv(hiddin, hiddin))
            self.bns.append(torch.nn.BatchNorm1d(hiddin))
        self.dropout = dropout

    def forward(self, X0, Y0, edge_index):
        # set initial values of ODEs
        X = X0
        Y = Y0
        # solve ODEs using simple IMEX scheme
        for i in range(self.num_layers):
            Y = Y + self.dt * (torch.relu(self.convs[i](X, edge_index)) - self.alpha * Y - self.gamma * X)
            X = X + self.dt * Y

            if (self.dropout is not None):
                Y = F.dropout(Y, self.dropout, training=self.training)
                X = F.dropout(X, self.dropout, training=self.training)
            X = self.bns[i](self.convs[i](X, edge_index))

        return X


# from torch_geometric.nn import GCNConv
# class deep_GNN(nn.Module):
#     '''
#     A deep GraphCON model using for instance Kipf & Wellings GCN
#     '''
#     def __init__(self, nfeat, nhid, nclass, nlayers, dt=1., alpha=1., gamma=1., dropout=None):
#         super(deep_GNN, self).__init__()
#         self.enc = nn.Linear(nfeat, nhid)
#         self.GNNs = nn.ModuleList()
#         for _ in range(nlayers):
#             self.GNNs.append(GCNConv(nhid, nhid))
#         self.graphcon = GraphCON(self.GNNs, dt, alpha, gamma, dropout)
#         self.dec = nn.Linear(nhid, nclass)
#
#     def forward(self, x, edge_index):
#         # compute initial values of ODEs (encode input)
#         X0 = self.enc(x)
#         Y0 = X0
#         # stack GNNs using GraphCON
#         X, Y = self.graphcon(X0, Y0, edge_index)
#         # decode X state of GraphCON at final time for output nodes
#         output = self.dec(X)
#         return output


class ONGNNConv(MessagePassing):
    def __init__(self, hidden_channel, chunk_size=32):
        '''
        The network implementation is from the paper "Ordered GNN:
            Ordering Message Passing to Deal with Heterophily and Over-smoothing"
        <https://arxiv.org/abs/2302.01524>

        this network requires a linear layer after the output to obtain valuable results,
        and when applying it to graph-level tasks, it needs other GNNs to assist in feature extraction.

        '''
        super(ONGNNConv, self).__init__('mean')

        self.add_self_loops = False
        self.tm = True
        self.simple_gating = False
        self.diff_or = True
        self.hidden_channel = hidden_channel
        self.chunk_size = chunk_size
        self.tm_net = Linear(2 * self.hidden_channel, self.chunk_size)
        # self.tm_norm = LayerNorm(self.hidden_channel)

    def reset_parameters(self):
        self.tm_net.reset_parameters()

    def forward(self, x, edge_index, last_tm_signal):
        if isinstance(edge_index, SparseTensor):
            edge_index = fill_diag(edge_index, fill_value=0)
            if self.add_self_loops == True:
                edge_index = fill_diag(edge_index, fill_value=1)
        else:
            edge_index, _ = remove_self_loops(edge_index)
            if self.add_self_loops == True:
                edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        m = self.propagate(edge_index, x=x)
        if self.tm == True:
            if self.simple_gating == True:
                tm_signal_raw = F.sigmoid(self.tm_net(torch.cat((x, m), dim=1)))
            else:
                tm_signal_raw = F.softmax(self.tm_net(torch.cat((x, m), dim=1)), dim=-1)
                tm_signal_raw = torch.cumsum(tm_signal_raw, dim=-1)
                if self.diff_or == True:
                    tm_signal_raw = last_tm_signal + (1 - last_tm_signal) * tm_signal_raw
            tm_signal = tm_signal_raw.repeat_interleave(repeats=int(self.hidden_channel / self.chunk_size), dim=1)
            tm_signal = F.pad(tm_signal, pad=(0, x.size(-1) - tm_signal.size(-1)), mode='constant', value=1)
            out = x * tm_signal + m * (1 - tm_signal)
        else:
            out = m
            tm_signal_raw = last_tm_signal

        # out = self.tm_norm(out)

        return out, tm_signal_raw


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    fill_value = 2. if improved else 1.
    num_nodes = int(edge_index.max()) + 1 if num_nodes is None else num_nodes
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class DAGNN(MessagePassing):
    '''
    The network implementation is from the paper "Towards
        Deeper Graph Neural Networks"
    <https://arxiv.org/abs/2007.09296>

    '''

    def __init__(self, num_classes, K=12, bias=True, **kwargs):
        super(DAGNN, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.proj = Linear(num_classes, 1)

    def forward(self, x, edge_index, edge_weight=None):
        # edge_index, norm = GCNConv.norm(edge_index, x.size(0), edge_weight, dtype=x.dtype)
        edge_index, norm = gcn_norm(edge_index, edge_weight, x.size(0), dtype=x.dtype)

        preds = []
        preds.append(x)
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            preds.append(x)

        pps = torch.stack(preds, dim=1)
        retain_score = self.proj(pps)
        retain_score = retain_score.squeeze()
        retain_score = torch.sigmoid(retain_score)
        retain_score = retain_score.unsqueeze(1)
        out = torch.matmul(retain_score, pps).squeeze()
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)

    def reset_parameters(self):
        self.proj.reset_parameters()


class DeeperGCN(torch.nn.Module):
    '''
    The network implementation is from the paper "DeeperGCN:
        All You Need to Train Deeper GCNs"
    <https://arxiv.org/abs/2006.07739>

    '''

    def __init__(self, hidden_channels, num_layers):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = nn.LayerNorm(hidden_channels, elementwise_affine=True)
            act = nn.ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

    def forward(self, x, edge_index):
        x = self.layers[0].conv(x, edge_index)

        for layer in self.layers[1:]:
            x = layer(x, edge_index)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        return x

    def reset_parameters(self):
        for layers in self.layers:
            layers.reset_parameters()
