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
                 normalize: bool = True, K: int = 4, gamma: float = 0.75,
                 fea_drop: bool = 'normal', hop_att: bool = True,
                 heads: int = 1, base_model: str = 'gat', negative_slope=0.2,
                 edge_dim=None, concat=False, dropout=0.0,
                 use_ffN = False, diff_alpha=False, norm_type=None,**kwargs):

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
        self.diff_alpha = diff_alpha #use different alpha
        self.norm_type = norm_type

        if norm_type=="layer":
            self.norm = nn.LayerNorm(heads * out_channels) if self.concat else nn.LayerNorm(out_channels)
        elif norm_type=="batch":
            self.norm = nn.BatchNorm1d(heads * out_channels) if self.concat else nn.BatchNorm1d(out_channels)
        else:
            self.norm = None

        self.lin = Linear(in_channels, out_channels, weight_initializer='glorot')
        self.lin_2 = Linear(in_channels, out_channels, weight_initializer='glorot')
        if concat:
            self.lin_2 = Linear(in_channels, heads * out_channels, weight_initializer='glorot')

        if self.use_ffN:
            self.feed_forward = FFN(heads * out_channels, heads * out_channels, dropout*2) \
                if self.concat else FFN(out_channels, out_channels, dropout*2)


        if self.hop_att:
            self.q = torch.nn.Linear(in_channels, heads * out_channels, bias=False)
            self.k = torch.nn.Linear(in_channels, heads * out_channels, bias=False)
            self.lin = Linear(in_channels, heads * out_channels, weight_initializer='glorot')

        if base_model == 'gat':
            if self.diff_alpha:
                self.att_src = Parameter(torch.Tensor(K, 1, heads, in_channels))
                self.att_dst = Parameter(torch.Tensor(K, 1, heads, in_channels))
            else:
                self.att_src = Parameter(torch.Tensor(1, heads, in_channels))
                self.att_dst = Parameter(torch.Tensor(1, heads, in_channels))

            if edge_dim is not None:
                self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False, weight_initializer='glorot')
                self.att_edge = Parameter(torch.empty(1, heads, out_channels))
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
            glorot(self.att_dst)
            glorot(self.att_edge)
        if self.use_ffN:
            self.feed_forward.reset_parameters()
        if self.norm is not None:
            self.norm.reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None, x0: OptTensor = None,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

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

            self.d0, self.de0, del1 = 0, 0, None
            for k in range(self.K):
                x_ = self.propagate(edge_index, x=x, edge_weight=edge_weight, del1=del1, alpha=None)
                preds = preds * self.gamma
                x = x_ if self.fea_drop == 'simple' else x_ + preds.sum(dim=0)
                preds = torch.cat([preds, torch.unsqueeze(x_, dim=0)], dim=0)
                del1 = self.d0
            preds[0] = preds[0] / deg.view(-1, 1)

            if self.hop_att:
                K, H, C = self.K + 1, self.heads, self.out_channels
                q = self.q(preds).view(K, -1, H, C)
                k = self.k(preds).view(K, -1, H, C)
                q = q.permute(1, 2, 0, 3)
                k = k.permute(1, 2, 0, 3)
                att = torch.einsum('nhqc,nhkc->nhqk', q, k)
                att = F.softmax(att, dim=-1)
                att = F.dropout(att, p=self.dropout, training=self.training)
                v = self.lin(preds).view(K, -1, H, C)
                v = v.permute(1, 2, 0, 3)
                preds = torch.einsum('nhqk,nhkc->nhqc', att, v).mean(dim=-2)
                preds = preds.view(-1, H * C) if self.concat else preds.mean(dim=-2)
                preds = preds + x_t

        elif self.base_model == 'gat':
            preds = torch.unsqueeze(x, dim=0)

            if isinstance(edge_index, Tensor):
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                deg = degree(edge_index[0], x.size(0), dtype=x.dtype) + 1

            elif isinstance(edge_index, SparseTensor):
                edge_index = remove_diag(edge_index)
                deg = sparsesum(edge_index, dim=1) + 1

            self.preprocess_edge_attr(edge_attr)


            self.d0, self.de0, del1 = 0, 0, None
            for k in range(self.K):
                if self.diff_alpha or k == 0:
                    alpha = self.alpha_edge(x, edge_index, k)
                x_ = self.propagate(edge_index, x=x, alpha=alpha, edge_weight=None, del1=del1)
                preds = preds * self.gamma
                if self.fea_drop == 'simple':
                    x = x_
                else:
                    x = x_ + preds.sum(dim=0)
                preds = torch.cat([preds, torch.unsqueeze(x_, dim=0)], dim=0)
                del1 = self.d0

            preds[0] = preds[0] / deg.view(-1, 1)

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
            if x0 is not None:
                temp_param = 0.2
                return temp_param * x_t + (1 - temp_param) * x0
            return x_t

        if not self.hop_att:
            preds = self.lin(preds.sum(0)) + x_t

        if self.use_ffN:
            preds = self.feed_forward(preds)

        if self.norm is not None:
            preds = self.norm(preds)

        if x0 is not None:
            temp_param = 0.5
            return temp_param * preds + (1 - temp_param) * x0

        return preds

    def preprocess_edge_attr(self, edge_attr):
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            # self.e = edge_attr

            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            self.cached_alpha_edge = alpha_edge

        else:
            self.cached_alpha_edge = 0
            # self.e = None

    def alpha_edge(self, x, edge_index, K_layer):
        if self.diff_alpha:
            alpha_src = (x.unsqueeze(dim=1) * self.att_src[K_layer, :]).sum(dim=-1)  # Temporarily only supports ij and ji equal
            alpha_dst = (x.unsqueeze(dim=1) * self.att_dst[K_layer, :]).sum(dim=-1)
        else:
            alpha_src = (x.unsqueeze(dim=1) * self.att_src).sum(dim=-1)
            alpha_dst = (x.unsqueeze(dim=1) * self.att_dst).sum(dim=-1)

        alpha = (alpha_src, alpha_dst)

        alpha = self.edge_updater(edge_index, alpha=alpha)
        return alpha

    def message(self, x_i, x_j, edge_weight, del1, alpha):

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

        elif self.base_model == 'gat':
            c_ij, c_ji = alpha

            c_ij= c_ij.unsqueeze(-1).mean(dim=1)# heads
            c_ji = c_ij.unsqueeze(-1).mean(dim=1)

            if self.fea_drop == 'normal':
                with torch.no_grad():
                    self.d0 = (x_i * self.gamma + x_j * c_ij - self.de0) * c_ji  # *c_jj
                    if del1 is not None:
                        self.de0 = (x_i * c_ji + x_j * self.gamma - del1) * c_ij  # *c_ii
                    else:
                        self.de0 = (x_i * c_ji + x_j * self.gamma) * c_ij

            return x_j * c_ij

    def edge_update(self, alpha_j, alpha_i, edge_index_i, edge_index_j, ptr, size_i):
        if alpha_i is not None:
            alpha = alpha_j + alpha_i
        else:
            alpha = alpha_j

        alpha = alpha + self.cached_alpha_edge # If there is an inconsistency,
        # it may be that the edge attribute does not remove self-loops

        alpha = F.leaky_relu(alpha, self.negative_slope)
        c_ij = softmax(alpha, edge_index_i, ptr, size_i)

        if self.fea_drop == 'normal':
            c_ji = softmax(alpha, edge_index_j, ptr, size_i)
        else:
            c_ij = F.dropout(c_ij, p=self.dropout, training=self.training)
            c_ji = c_ij

        return c_ij, c_ji

    def update(self, aggr_out, del1, edge_index_j, dim_size):

        if self.fea_drop == 'normal' and del1 is not None:  # Paper formula 14
            aggr_out = aggr_out - self.aggregate(del1, edge_index_j, dim_size=dim_size)
        return aggr_out

