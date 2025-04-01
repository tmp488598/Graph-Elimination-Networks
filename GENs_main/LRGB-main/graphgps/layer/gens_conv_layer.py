import torch.nn as nn
from torch_geometric.graphgym import cfg
import torch_geometric.graphgym.register as register

from .models import GENsConv


class GENConvLayer(nn.Module):
    """
    """
    def __init__(self, dim_in, dim_out, dropout, residual):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual
        self.batch_norm = nn.BatchNorm1d(dim_out) if cfg.gnn.batchnorm else None
        param = cfg.gnn.gens

        self.init_res = cfg.gnn.init_res
        self.init_res_ratio = cfg.gnn.init_res_ratio

        self.act = nn.Sequential(
            register.act_dict[cfg.gnn.act](),
            nn.Dropout(self.dropout),
        )
        if param.concat:
            dim_out = int(dim_out/param.heads)

        self.model = GENsConv(dim_in, dim_out, edge_dim=param.edge_dim, hop_att=param.hop_att, K=param.K, gamma=param.gamma,
                                  fea_drop=param.fea_drop, base_model=param.base_model, concat=param.concat, heads=param.heads,
                                  dropout = param.att_dropout, use_ffN =param.use_ffN, diff_alpha = param.diff_alpha,
                                  norm_type=param.norm_type)

        self.use_edge = True if param.edge_dim is not None else False

    def forward(self, batch, x0=None):
        x_in = batch.x

        x_ini = x0 if self.init_res else None

        if self.use_edge:
            batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr, x0=x_ini, init_ratio=self.init_res_ratio)

        else:
            batch.x = self.model(batch.x, batch.edge_index, x0=x_ini, init_ratio=self.init_res_ratio)

        batch.x = self.act(batch.x)

        if self.batch_norm is not None:
            batch.x = self.batch_norm(batch.x)

        if self.residual:
            batch.x = x_in + batch.x  # residual connection

        return batch
