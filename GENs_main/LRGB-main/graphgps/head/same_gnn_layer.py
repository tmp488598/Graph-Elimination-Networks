import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import new_layer_config, MLP
from torch_geometric.graphgym.register import register_head
import torch_geometric.graphgym.register as register


@register_head('same_gnn')
class SAMEGNNHead(nn.Module):
    """
    Use the corresponding GNN layer as the task head, with only one layer

    Args:
        model: Output GNN layer.
    """

    def __init__(self, model):
        super(SAMEGNNHead, self).__init__()
        self.layer_post_mp = model
        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]

    def _apply_index(self, batch):
        return batch.x, batch.y

    def forward(self, batch):
        if self.layer_post_mp.lin_edge is not None:
            batch.x = self.layer_post_mp(batch.x, batch.edge_index, batch.edge_attr)
        else:
            batch.x = self.layer_post_mp(batch.x, batch.edge_index)

        if batch.x.size(0) != batch.y.size(0):
            batch.x = self.pooling_fun(batch.x, batch.batch)

        pred, label = self._apply_index(batch)
        return pred, label
