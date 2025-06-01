import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head
from torch_geometric.graphgym.models.layer import new_layer_config, MLP
from graphgps.layer.neuralwalker_layer import WalkEncoder

@register_head('global_head')
class GNNWalkNodeHead(nn.Module):
    """
    GNN prediction head incorporating WalkEncoder for inductive node prediction tasks.

    Args:
        dim_in (int): Input dimension (dimension of node features after GNN).
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out):
        super(GNNWalkNodeHead, self).__init__()
        self.walk_encoder = WalkEncoder(dim_in, 'transformer', num_heads=4)

        self.layer_post_mp = MLP(
            new_layer_config(dim_in, dim_out, cfg.gnn.layers_post_mp,
                             has_act=False, has_bias=True, cfg=cfg))


    def _apply_index(self, batch):
        return batch.x, batch.y

    def forward(self, batch):

        batch.walk_pe = torch.cat([batch.walk_node_id_encoding, batch.walk_node_adj_encoding], dim=-1)

        batch = self.walk_encoder(batch)

        batch = self.layer_post_mp(batch)
        pred, label = self._apply_index(batch)

        return pred, label

