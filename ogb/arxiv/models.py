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
from torch_sparse import  remove_diag,  matmul, mul
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
    def __init__(self, nfeat, nhid, dropout, num_hops,bn=True):
        super(DGMLP, self).__init__()
        self.bn = bn
        self.num_layers = num_hops
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid))
        for i in range(self.num_layers-1):
            self.convs.append(GCNConv(nhid, nhid))
            self.bns.append(torch.nn.BatchNorm1d(nhid))
        self.dropout = dropout
        self.lr_att = nn.Linear(nhid+nhid, 1)

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
            x = (1-alpha)*x+(alpha)*x_input
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.bns[i-1](self.convs[i](x, adj)) if self.bn else self.convs[i](x, adj)
        return x


class GraphCON(nn.Module):
    '''
    The network implementation is from the paper "Graph-Coupled
        Oscillator Networks"
    <https://arxiv.org/abs/2202.02296>
    '''
    def __init__(self, GNNs, dt=1., alpha=1., gamma=1., dropout=None):
        super(GraphCON, self).__init__()
        self.dt = dt
        self.alpha = alpha
        self.gamma = gamma
        self.GNNs = GNNs  # list of the individual GNN layers
        self.dropout = dropout

    def forward(self, X0, Y0, edge_index):
        # set initial values of ODEs
        X = X0
        Y = Y0
        # solve ODEs using simple IMEX scheme
        for gnn in self.GNNs:
            Y = Y + self.dt * (torch.relu(gnn(X, edge_index)) - self.alpha * Y - self.gamma * X)
            X = X + self.dt * Y

            if (self.dropout is not None):
                Y = F.dropout(Y, self.dropout, training=self.training)
                X = F.dropout(X, self.dropout, training=self.training)

        return X, Y

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
    def __init__(self, hidden_channel, chunk_size):
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
        self.tm_net = Linear(2*self.hidden_channel, self.chunk_size)
        # self.tm_norm = LayerNorm(self.hidden_channel)

    def reset_parameters(self):
        self.tm_net.reset_parameters()


    def forward(self, x, edge_index, last_tm_signal):
        if isinstance(edge_index, SparseTensor):
            edge_index = fill_diag(edge_index, fill_value=0)
            if self.add_self_loops==True:
                edge_index = fill_diag(edge_index, fill_value=1)
        else:
            edge_index, _ = remove_self_loops(edge_index)
            if self.add_self_loops==True:
                edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        m = self.propagate(edge_index, x=x)
        if self.tm==True:
            if self.simple_gating==True:
                tm_signal_raw = F.sigmoid(self.tm_net(torch.cat((x, m), dim=1)))
            else:
                tm_signal_raw = F.softmax(self.tm_net(torch.cat((x, m), dim=1)), dim=-1)
                tm_signal_raw = torch.cumsum(tm_signal_raw, dim=-1)
                if self.diff_or==True:
                    tm_signal_raw = last_tm_signal+(1-last_tm_signal)*tm_signal_raw
            tm_signal = tm_signal_raw.repeat_interleave(repeats=int(self.hidden_channel/self.chunk_size), dim=1)
            tm_signal = F.pad(tm_signal, pad=(0, x.size(-1)-tm_signal.size(-1)), mode='constant', value=1)
            out = x*tm_signal + m*(1-tm_signal)
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
    def __init__(self, num_classes, K, bias=True, **kwargs):
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


def gen_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=False, dtype=None):
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        adj_t = remove_diag(adj_t)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t, deg

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col], deg


class GENsConv(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]
    '''
    
    Args:
        K: number of hops for single-layer network propagation.
        gamma：a hyperparameter that controls the decay, used to balance 
            the difference between the central feature and other features in 
            the receptive field. When K is greater than 2, it can be considered
            to reduce it appropriately.
        fea_drop: elimination method, supports normal/simple/None, corresponding 
            to normal elimination, removing self-loops, or doing nothing.
        hop_att: whether to use self-attention across multiple hops.
        heads: number of attention heads.
        base_model: base model, supports gcn and gat.
        x0: used to pass the initial features of the dataset, increase the initial
            residual, can be considered when the number of stacked layers is very deep.
    '''
    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 normalize: bool = True, K: int = 8, gamma: float = 0.85,
                 fea_drop: bool = 'simple', hop_att: bool = False,
                 heads: int = 1, base_model: str = 'gcn', negative_slope = 0.2,
                 edge_dim=None, concat=False, dropout=0, **kwargs):


        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.K = K
        self.d0 = None
        self.de0 = None
        self.gamma = gamma
        self.fea_drop = fea_drop
        self.hop_att = hop_att
        self.heads = heads
        self.base_model = base_model
        self._cached_edge_index = None
        self._cached_adj_t = None
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope


        self.lin = Linear(in_channels, out_channels, weight_initializer='glorot')
        self.lin_2 = Linear(in_channels, out_channels, weight_initializer='glorot')

        if self.hop_att:
            self.q = torch.nn.Linear(in_channels, self.heads * out_channels, bias=False)
            self.k = torch.nn.Linear(in_channels, self.heads * out_channels, bias=False)
            self.lin = Linear(in_channels, self.heads * out_channels, weight_initializer='glorot')
            if self.concat:
                self.lin_2 = Linear(in_channels, out_channels * heads, weight_initializer='glorot')

        if base_model == 'gat':
            self.att_src = Parameter(torch.Tensor(1, heads, in_channels))
            if edge_dim is not None:
                self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False, weight_initializer='glorot')
                self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
            else:
                self.lin_edge = None
                self.register_parameter('att_edge', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.lin_2.reset_parameters()
        if self.hop_att:
            self.q.reset_parameters()
            self.k.reset_parameters()
        if self.base_model == 'gat':
            if self.lin_edge is not None:
                self.lin_edge.reset_parameters()
            glorot(self.att_src)
            glorot(self.att_edge)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj, x0: OptTensor= None,
                edge_weight: OptTensor = None, edge_attr: OptTensor = None, deg=None) -> Tensor:
        """"""
        
        # x_t = self.lin_2(x*self.gamma**self.K)
        x_t = self.lin_2(x)

        if self.base_model == 'gcn':
            if self.normalize:
                if isinstance(edge_index, Tensor):
                    cache = self._cached_edge_index
                    if cache is None:
                        edge_index, edge_weight, deg = gen_norm( 
                            edge_index, edge_weight, x.size(self.node_dim),
                            self.improved)
                        if self.cached:
                            self._cached_edge_index = (edge_index, edge_weight, deg)
                    else:
                        edge_index, edge_weight, deg = cache[0], cache[1], cache[2]

                elif isinstance(edge_index, SparseTensor):
                    cache = self._cached_adj_t
                    if cache is None:
                        edge_index, deg = gen_norm(
                            edge_index, edge_weight, x.size(self.node_dim),
                            self.improved)
                        if self.cached:
                            self._cached_adj_t = (edge_index, deg)
                    else:
                        edge_index, deg = cache[0], cache[1]

            preds = torch.unsqueeze(x, dim=0)

            if isinstance(edge_index, SparseTensor):  # Sparse only supports 'simple'
                self.fea_drop = 'simple'

            self.d0, self.de0, del1 = 0, 0, None
            for k in range(self.K):
                x_ = self.propagate(edge_index, x=x, edge_weight=edge_weight, del1=del1, alpha=None, edge_attr=None,
                                    num_node=x.size(0), deg=deg)
                preds = preds * self.gamma
                x = x_ if self.fea_drop=='simple' else x_ + preds.sum(dim=0)
                preds = torch.cat([preds, torch.unsqueeze(x_, dim=0)], dim=0)
                del1 = self.d0
            preds[0] = preds[0] / deg.view(-1, 1)

        elif self.base_model == 'gat':
            if isinstance(edge_index, SparseTensor):
                edge_index = torch.stack(edge_index.coo()[:2], dim=0)

            preds = torch.unsqueeze(x, dim=0)
            alpha = (x.unsqueeze(dim=1) * self.att_src).sum(dim=-1)  # Temporarily only supports ij and ji equal
            edge_index, _ = remove_self_loops(edge_index)

            deg = degree(edge_index[0], x.size(0), dtype=x.dtype) + 1

            self.d0, self.de0, del1 = 0, 0, None
            for k in range(self.K):
                x_ = self.propagate(edge_index, x=x, alpha=alpha, edge_attr=edge_attr, edge_weight=None,
                                    num_node=x.size(0), del1=del1, deg=deg)
                preds = preds * self.gamma 
                x = x_ if self.fea_drop=='simple' else x_ + preds.sum(dim=0)
                preds = torch.cat([preds, torch.unsqueeze(x_, dim=0)], dim=0)
                del1 = self.d0

            preds[0] = preds[0] / deg.view(-1, 1)

        else:
            if x0 is not None:
                temp_param = 0.2
                return temp_param * x_t + (1 - temp_param) * x0
            return x_t

        if self.hop_att:
            K, H, C = self.K + 1, self.heads, self.out_channels
            q = self.q(preds).view(K, -1, H, C)  # Multi-head self-attention
            k = torch.unsqueeze(self.k(preds[0]).view(-1, H, C), dim=-2)
            att = torch.einsum('nxhd,xhyd->xnhy', [q, k])
            att = F.softmax(att, dim=1)
            preds = self.lin(preds).view(K, -1, H, C).transpose(0, 1) * att
            preds = preds.sum(dim=1).view(-1,H*C) if self.concat else preds.sum(dim=1).mean(dim=-2)
            out = preds + x_t

        else:
            out = self.lin(preds.sum(0)) + x_t

        if x0 is not None:
            temp_param = 0.2
            return temp_param*out+(1-temp_param)*x0

        return out

    def message(self, x_i, x_j, edge_weight, del1, alpha, edge_attr, edge_index, num_node, deg) -> Tensor:
        if self.base_model == 'gcn':
            edge_weight = 1 if edge_weight is None else edge_weight.view(-1, 1)

            if self.fea_drop =='normal':# Paper formula 13
                with torch.no_grad(): 
                    self.d0 = (x_i * self.gamma + x_j * edge_weight - self.de0) * edge_weight  # *c_jj
                    if del1 is not None:
                        self.de0 = (x_i * edge_weight + x_j * self.gamma - del1) * edge_weight  # *c_ii
                    else:
                        self.de0 = (x_i * edge_weight + x_j * self.gamma) * edge_weight
            return edge_weight * x_j
        else:
            alpha = torch.index_select(alpha, dim=0, index=edge_index[0]) + torch.index_select(alpha, dim=0, index=edge_index[1])

            if edge_attr is not None:
                if edge_attr.dim() == 1:
                    edge_attr = edge_attr.view(-1, 1)
                assert self.lin_edge is not None
                edge_attr = self.lin_edge(edge_attr)
                edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
                alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
                alpha = alpha + alpha_edge

            c_ij = c_ji = alpha

            c_ij = F.leaky_relu(c_ij, self.negative_slope)
            c_ij = softmax(c_ij, edge_index[1], num_nodes=num_node)
            c_ij = F.dropout(c_ij, p=self.dropout, training=self.training).unsqueeze(-1).mean(dim=1)

            if self.fea_drop =='normal':
                c_ji = F.leaky_relu(c_ji, self.negative_slope)
                c_ji = softmax(c_ji, edge_index[0], num_nodes=num_node)
                c_ji = F.dropout(c_ji, p=self.dropout, training=self.training).unsqueeze(-1).mean(dim=1)
                with torch.no_grad():
                    self.d0 = (x_i * self.gamma + x_j * c_ij - self.de0) * c_ji  # *c_jj
                    if del1 is not None:
                        self.de0 = (x_i * c_ji + x_j * self.gamma - del1) * c_ij  # *c_ii
                    else:
                        self.de0 = (x_i * c_ji + x_j * self.gamma) * c_ij

            # return x_j * c_ij * torch.index_select(1 - 1 / deg, dim=0, index=edge_index[0]).view(-1, 1)
            return x_j * c_ij

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def update(self, aggr_out, del1, edge_index, dim_size):

        if self.fea_drop =='normal' and del1 is not None:# Paper formula 14
            aggr_out = aggr_out - self.aggregate(del1, edge_index[0], dim_size=dim_size)
        return aggr_out