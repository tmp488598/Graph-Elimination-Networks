import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import global_mean_pool, global_add_pool
from models import MLP
from MAD_metric import mad_value
from utils import gnn_layer_dict


class GNN(torch.nn.Module):

    def __init__(self, net_params):
        super(GNN, self).__init__()

        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        self.dropout = net_params['dropout']
        n_layers = net_params['num_layers']
        self.num_fc = net_params['num_fc']
        global_pool = net_params['global_pool']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.cat = net_params['concat']
        self.node_encoder = net_params['node_encoder']
        self.sum_x = net_params['sum_x']
        self.use_x0 = True if net_params['gnn_type'] in['gcn2','gcon'] else False
        self.use_last_signal = True if net_params['gnn_type'] in ['ongnn'] else False

        self.edge_dim = net_params["gens_edge_dim"]  if net_params["gens_edge_dim"] > 0 else None

        if "add" in  global_pool:
            self.global_pool = global_add_pool
        else:
            self.global_pool = global_mean_pool

        if 'virtual' in global_pool:
            self.vnode_pooling = True
        else:
            self.vnode_pooling = False

        if net_params['use_xg']:  # Utilize graph level features.
            self.use_xg = True
            self.bn1_xg = BatchNorm1d(net_params['xg_size'])
            self.lin1_xg = Linear(net_params['xg_size'], hidden_dim)
            self.bn2_xg = BatchNorm1d(hidden_dim)
            self.lin2_xg = Linear(hidden_dim, hidden_dim)
        else:
            self.use_xg = False

        if self.node_encoder > 0:
            self.lin_encoder = MLP(in_dim, hidden_dim, hidden_dim, self.node_encoder, dropout=self.dropout)
            in_dim = hidden_dim

        self.edge_en = MLP(4, hidden_dim, hidden_dim, self.node_encoder, dropout=self.dropout)



        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        gnn_conv = gnn_layer_dict.get(net_params['gnn_type'])

        for i in range(n_layers):
            if 'gens' in net_params['gnn_type']:
                if net_params["gens_att_concat"]:
                    self.convs.append(
                        gnn_conv(in_dim, hidden_dim//net_params["gens_heads"], K=net_params["gens_K"], gamma=net_params["gens_gamma"],
                                 fea_drop=net_params["gens_fea_drop"],
                                 hop_att=net_params["gens_hop_att"], heads=net_params["gens_heads"],
                                 base_model=net_params["gens_base_model"],
                                 edge_dim=self.edge_dim, concat=net_params["gens_att_concat"],
                                 dropout=net_params["gens_att_dropout"]))
                else:
                    self.convs.append(gnn_conv(in_dim, hidden_dim, K=net_params["gens_K"], gamma=net_params["gens_gamma"], fea_drop=net_params["gens_fea_drop"],
                                           hop_att=net_params["gens_hop_att"], heads=net_params["gens_heads"], base_model=net_params["gens_base_model"],
                                           edge_dim = self.edge_dim, concat=net_params["gens_att_concat"], dropout=net_params["gens_att_dropout"]))
            elif self.use_x0 or self.use_last_signal:
                self.convs.append(gnn_conv(hidden_dim))
            elif 'gin' in net_params['gnn_type']:
                self.convs.append(gnn_conv(Linear(in_dim, hidden_dim), train_eps=True))
            elif 'dagnn' in net_params['gnn_type']:
                self.convs.append(gnn_conv(hidden_dim))
            else:
                self.convs.append(gnn_conv(in_dim, hidden_dim))
            self.bns_conv.append(BatchNorm1d(in_dim))
            in_dim = hidden_dim

        # self.bn_hidden = BatchNorm1d(hidden_dim)
        if self.cat:
            in_dim = hidden_dim*n_layers
            self.bn_hidden = BatchNorm1d(hidden_dim)

        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        for i in range(self.num_fc - 1):
            self.bns_fc.append(BatchNorm1d(in_dim))
            self.lins.append(Linear(in_dim, hidden_dim))
            in_dim=hidden_dim

        self.lin_class = Linear(in_dim, out_dim)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.edge_dim is not None:
            edge_attr = self.edge_en(data.edge_attr)

        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        x = self.lin_encoder(x) if self.node_encoder > 0 else x
        if self.use_last_signal:
            tm_signal = x.new_zeros(32)

        xs = [x]
        for conv, batch_norm in zip(self.convs, self.bns_conv):
            x = batch_norm(xs[-1]) if self.batch_norm else xs[-1]
            if self.use_x0:
                x = conv(x, xs[0], edge_index)
            elif self.use_last_signal:
                x, tm_signal = conv(x, edge_index, last_tm_signal=tm_signal)
            else:
                x = conv(x, edge_index, edge_attr = edge_attr)  if self.edge_dim is not None else conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + xs[-1] if self.residual and len(xs)>1 else x
            xs.append(x)

        if self.sum_x:
            out_x = 0
            for sx in xs[1:]:
                out_x += sx
            xs[-1] = out_x

        x = torch.cat(xs[1:], dim=-1) if self.cat else xs[-1]

        x = self.global_pool(x, batch) if not self.vnode_pooling else x[self.find_virtual_nodes(batch)]
        x = x if xg is None else x + xg

        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x_

        # x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.num_fc > 0:
            x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def forward_node_mad(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.lin_encoder(x) if self.node_encoder > 0 else x

        xs = [x]
        for conv, batch_norm in zip(self.convs, self.bns_conv):
            x = batch_norm(xs[-1]) if self.batch_norm else xs[-1]
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + xs[-1] if self.residual and len(xs) > 1 else x
            xs.append(x)

        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]

        mad_values = []
        # Compute MAD for each graph in the batch
        for graph_idx in torch.unique(batch):
            mask = (batch == graph_idx)
            graph_x = x[mask]
            graph_mad = mad_value(graph_x)
            mad_values.append(graph_mad)
        # Compute the average of MAD values across all graphs
        average_mad = torch.tensor(mad_values).mean()

        return average_mad

    def find_virtual_nodes(self, batch):
        cum_size = torch.bincount(batch)
        shifts = torch.cumsum(cum_size, dim=0)
        virtual_node_indices = shifts - 1
        return virtual_node_indices

    def __repr__(self):
        return self.__class__.__name__