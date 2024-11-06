import torch.nn as nn
from torch_geometric.graphgym import cfg
import torch_geometric.graphgym.register as register

from .models import GENsConv


class GENConvLayer(nn.Module):
    """
    """
    def __init__(self, dim_in, dim_out, dropout, residual, param=None):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual

        self.act = nn.Sequential(
            register.act_dict[cfg.gnn.act](),
            nn.Dropout(self.dropout),
        )
        if param.concat:
            self.model = GENsConv(dim_in, int(dim_out/param.heads), edge_dim=param.edge_dim, hop_att=param.hop_att, K=param.K, gamma=param.gamma,
                              fea_drop=param.fea_drop, base_model=param.base_model,concat=param.concat, heads=param.heads,
                              dropout = param.att_dropout, use_ffN =param.use_ffN)
        else:
            self.model = GENsConv(dim_in, dim_out, edge_dim=param.edge_dim, hop_att=param.hop_att, K=param.K, gamma=param.gamma,
                                  fea_drop=param.fea_drop, base_model=param.base_model, concat=param.concat, heads=param.heads,
                                  dropout = param.att_dropout, use_ffN =param.use_ffN)
        self.use_edge = True if param.edge_dim is not None else False

    def forward(self, batch):
        x_in = batch.x
        if self.use_edge:
            batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        else:
            batch.x = self.model(batch.x, batch.edge_index)
        batch.x = self.act(batch.x)

        if self.residual:
            batch.x = x_in + batch.x  # residual connection

        return batch
