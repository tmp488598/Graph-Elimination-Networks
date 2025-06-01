import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from torch_sparse import SparseTensor, fill_diag
from typing import Optional, Tuple
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import degree, softmax
from torch_sparse import  remove_diag,  matmul, mul
from torch_sparse import sum as sparsesum
from torch_geometric.utils import is_undirected

class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels)
        self.fc2 = nn.Linear(in_channels, out_channels)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)


    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, x):
        x_ = self.ff_dropout1(F.relu(self.fc1(x)))
        x = self.ff_dropout2(self.fc2(x_)) + x
        return x



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
    concat: Whether to concatenate the outputs of different attention heads, note that this will expand the dimension.
    norm_type: The normalization type after the feed-forward network (FFN), supports 'layer', 'batch', or 'None'.
    no_param: Whether to run a non-parametric toy example.
    '''

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 normalize: bool = True, K: int = 1, gamma: float = 1,
                 fea_drop: bool = 'simple', hop_att: bool = False,
                 heads: int = 1, base_model: str = 'gcn', negative_slope=0.2,
                 edge_dim=None, concat=False, dropout=0.0,
                 use_ffN=False, norm_type=None,
                 no_param=False, **kwargs):

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
        self.use_ffN = use_ffN
        self.norm_type = norm_type
        self.no_param = no_param
        self.is_undirected =None

        if norm_type == "layer":
            self.norm = nn.LayerNorm(heads * out_channels) if self.concat else nn.LayerNorm(out_channels)
        elif norm_type == "batch":
            self.norm = nn.BatchNorm1d(heads * out_channels) if self.concat else nn.BatchNorm1d(out_channels)
        else:
            self.norm = None

        self.lin = Linear(in_channels, out_channels, weight_initializer='glorot')
        self.lin_2 = Linear(in_channels, out_channels, weight_initializer='glorot')
        if concat:
            self.lin_2 = Linear(in_channels, heads * out_channels, weight_initializer='glorot')
            self.lin = Linear(in_channels, heads * out_channels, weight_initializer='glorot')

        if self.use_ffN:
            self.feed_forward = FFN(heads * out_channels, heads * out_channels, dropout) \
                if self.concat else FFN(out_channels, out_channels, dropout)

        if self.hop_att:
            self.q = torch.nn.Linear(in_channels, heads * out_channels, bias=False)
            self.k = torch.nn.Linear(in_channels, heads * out_channels, bias=False)
            self.lin = Linear(in_channels, heads * out_channels, weight_initializer='glorot')


        if base_model == 'gat':
            self.att_src = Parameter(torch.Tensor(1, heads, in_channels))
            if edge_dim is not None:
                self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False, weight_initializer='glorot')
                self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
            else:
                self.lin_edge = None
                self.register_parameter('att_edge', None)
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
        if self.use_ffN:
            self.feed_forward.reset_parameters()
        if self.norm is not None:
            self.norm.reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None,
                edge_weight: OptTensor = None, x0: OptTensor = None,
                init_ratio=0.0) -> Tensor:
        """"""

        if self.is_undirected is None:
            self.is_undirected = is_undirected(edge_index)
            if not self.is_undirected:
                print("Input is directed graph!")

        if self.no_param:
            x_t = x
        else:
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
                # preds = preds * self.gamma
                x = x_ if self.fea_drop == 'simple' else x_ + preds.sum(dim=0)
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
                # preds = preds * self.gamma
                x = x_ if self.fea_drop == 'simple' else x_ + preds.sum(dim=0)
                preds = torch.cat([preds, torch.unsqueeze(x_, dim=0)], dim=0)
                del1 = self.d0

            preds[0] = preds[0] / deg.view(-1, 1)

        else:
            return (1-init_ratio) * x_t + init_ratio * x0


        def power_compression(x, gamma):
            eps = 1e-8
            norm = torch.linalg.norm(x, dim=-1, keepdim=True) + eps
            return x / norm.pow(gamma)

        if not self.no_param:
            preds = power_compression(preds, self.gamma)

        if self.hop_att:
            K, H, C = self.K + 1, self.heads, self.out_channels
            q = self.q(preds).view(K, -1, H, C)  # Multi-head self-attention
            k = torch.unsqueeze(self.k(preds[0]).view(-1, H, C), dim=-2)
            att = torch.einsum('nxhd,xhyd->xnhy', [q, k])
            att = F.softmax(att, dim=1)
            # att = F.dropout(att, p=self.dropout, training=self.training)
            preds = self.lin(preds).view(K, -1, H, C).transpose(0, 1) * att
            preds = preds.sum(dim=1).view(-1, H * C) if self.concat else preds.sum(dim=1).mean(dim=-2)
            preds = preds + x_t

        else:
            if self.no_param:
                preds = preds.sum(0) + x_t
            else:
                preds = self.lin(preds.sum(0)) + x_t

        if self.use_ffN:
            preds = self.feed_forward(preds)

        if self.norm is not None:
            preds = self.norm(preds)

        if x0 is not None:
            return (1-init_ratio) * preds + init_ratio * x0

        return preds

    def message(self, x_i, x_j, edge_weight, del1, alpha, edge_attr, edge_index, num_node, deg) -> Tensor:
        if self.base_model == 'gcn':
            edge_weight = 1 if edge_weight is None else edge_weight.view(-1, 1)

            if self.fea_drop == 'normal':  # Paper formula 13
                with torch.no_grad():
                    self.d0 = (x_i * 1 + x_j * edge_weight - self.de0) * edge_weight  # *c_jj
                    if del1 is not None:
                        self.de0 = (x_i * edge_weight + x_j * 1 - del1) * edge_weight  # *c_ii
                    else:
                        self.de0 = (x_i * edge_weight + x_j * 1) * edge_weight


            return edge_weight * x_j
        else:
            alpha = torch.index_select(alpha, dim=0, index=edge_index[0]) + torch.index_select(alpha, dim=0,
                                                                                               index=edge_index[1])

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
            c_ij = softmax(c_ij, edge_index[1], num_nodes=num_node).unsqueeze(-1).mean(dim=1)
            # c_ij = F.dropout(c_ij, p=self.dropout, training=self.training).unsqueeze(-1).mean(dim=1)

            if self.fea_drop == 'normal':
                c_ji = F.leaky_relu(c_ji, self.negative_slope)
                c_ji = softmax(c_ji, edge_index[0], num_nodes=num_node).unsqueeze(-1).mean(dim=1)
                # c_ji = F.dropout(c_ji, p=self.dropout, training=self.training).unsqueeze(-1).mean(dim=1)
                with torch.no_grad():
                    self.d0 = (x_i * 1 + x_j * c_ij - self.de0) * c_ji  # *c_jj
                    if del1 is not None:
                        self.de0 = (x_i * c_ji + x_j * 1 - del1) * c_ij  # *c_ii
                    else:
                        self.de0 = (x_i * c_ji + x_j * 1) * c_ij

            # return x_j * c_ij * torch.index_select(1 - 1 / deg, dim=0, index=edge_index[0]).view(-1, 1)
            return x_j * c_ij

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def update(self, aggr_out, del1, edge_index, dim_size):

        if self.fea_drop == 'normal' and del1 is not None:  # Paper formula 14

            aggr_out = (aggr_out - self.aggregate(del1, edge_index[0], dim_size=dim_size))

        return aggr_out



if __name__ == '__main__':
    model = GENsConv(5, 5, base_model='gcn', hop_att=False, fea_drop='normal', K=10, gamma=1.0, no_param=True)  # Adjusting the K test for the effect of GEA
    x = torch.tensor([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]],
                     dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 2, 3, 1, 2, 3,4], [1, 2, 3, 4, 0, 1, 2,3]])
    out = model(x, edge_index)
    print(out)