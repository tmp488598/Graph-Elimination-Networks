import torch
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import add_remaining_self_loops, scatter
from typing import Optional, Tuple
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import softmax, degree



def gen_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=False, dtype=None):
    fill_value = 2. if improved else 1.

    if True:
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
        deg = scatter(edge_weight, col, dim=0, dim_size=num_nodes, reduce='sum') + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col], deg


class GENsConv(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
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
                 normalize: bool = True, K: int = 4, gamma: float = 0.8,
                 fea_drop: bool = 'simple', hop_att: bool = True,
                 heads: int = 2, base_model: str = 'gat', negative_slope=0.2,
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
            self.q = torch.nn.Linear(in_channels, self.heads * out_channels)
            self.k = torch.nn.Linear(in_channels, self.heads * out_channels)
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

    def forward(self, x: Tensor, edge_index: Adj, x0: OptTensor = None,
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

            preds = torch.unsqueeze(x, dim=0)

            self.d0, self.de0, del1 = 0, 0, None
            for k in range(self.K):
                x_ = self.propagate(edge_index, x=x, edge_weight=edge_weight, del1=del1, alpha=None, edge_attr=None,
                                    num_node=x.size(0), deg=deg)
                preds = preds * self.gamma
                x = x_ if self.fea_drop == 'simple' else x_ + preds.sum(dim=0)
                preds = torch.cat([preds, torch.unsqueeze(x_, dim=0)], dim=0)
                del1 = self.d0
            preds[0] = preds[0] / deg.view(-1, 1)

        elif self.base_model == 'gat':

            preds = torch.unsqueeze(x, dim=0)
            alpha = (x.unsqueeze(dim=1) * self.att_src).sum(dim=-1)  # Temporarily only supports ij and ji equal
            edge_index, _ = remove_self_loops(edge_index)

            deg = degree(edge_index[0], x.size(0), dtype=x.dtype) + 1

            self.d0, self.de0, del1 = 0, 0, None
            for k in range(self.K):
                x_ = self.propagate(edge_index, x=x, alpha=alpha, edge_attr=edge_attr, edge_weight=None,
                                    num_node=x.size(0), del1=del1, deg=deg)
                preds = preds * self.gamma
                x = x_ if self.fea_drop == 'simple' else x_ + preds.sum(dim=0)
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
            preds = preds.sum(dim=1).view(-1, H * C) if self.concat else preds.sum(dim=1).mean(dim=-2)
            out = preds + x_t

        else:
            out = self.lin(preds.sum(0)) + x_t

        if x0 is not None:
            temp_param = 0.2
            return temp_param * out + (1 - temp_param) * x0

        return out

    def message(self, x_i, x_j, edge_weight, del1, alpha, edge_attr, edge_index, num_node, deg) -> Tensor:
        if self.base_model == 'gcn':
            edge_weight = 1 if edge_weight is None else edge_weight.view(-1, 1)

            if self.fea_drop == 'normal':  # Paper formula 13
                with torch.no_grad():
                    self.d0 = (x_i * self.gamma + x_j * edge_weight - self.de0) * edge_weight  # *c_jj
                    if del1 is not None:
                        self.de0 = (x_i * edge_weight + x_j * self.gamma - del1) * edge_weight  # *c_ii
                    else:
                        self.de0 = (x_i * edge_weight + x_j * self.gamma) * edge_weight
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
            c_ij = softmax(c_ij, edge_index[1], num_nodes=num_node)
            c_ij = F.dropout(c_ij, p=self.dropout, training=self.training).unsqueeze(-1).mean(dim=1)

            if self.fea_drop == 'normal':
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

    def update(self, aggr_out, del1, edge_index, dim_size):

        if self.fea_drop == 'normal' and del1 is not None:  # Paper formula 14
            aggr_out = aggr_out - self.aggregate(del1, edge_index[0], dim_size=dim_size)
        return aggr_out