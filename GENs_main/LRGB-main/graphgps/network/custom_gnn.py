import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder
from torch_geometric.graphgym.register import register_network

from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvLayer
from graphgps.layer.gcn_conv_layer import GCNConvLayer
from graphgps.layer.gens_conv_layer import GENConvLayer
from graphgps.layer.pre_gnn_layer import GNNPreMP
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from graphgps.layer.models import GENsConv


@register_network('custom_gnn')
class CustomGNN(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in
        dim_out = cfg.gnn.dim_out if cfg.gnn.dim_out is not None else dim_out

        conv_model = self.build_conv_model(cfg.gnn.layer_type)

        layers_mp = cfg.gnn.layers_mp
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner,
                                   cfg.gnn.layers_pre_mp,gnn_type = cfg.gnn.pre_gnn_type)
        else:
            self.pre_mp = conv_model(dim_in,
                                     cfg.gnn.dim_inner,
                                     dropout=cfg.gnn.dropout,
                                     residual=cfg.gnn.residual)
            layers_mp -= 1

        dim_in = cfg.gnn.dim_inner
        assert cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        self.gnn_layers = torch.nn.ModuleList()

        GNNHead = register.head_dict[cfg.gnn.head]
        if cfg.gnn.head == 'same_gnn':
            layers_mp -= 1
            self.post_mp = GNNHead(conv_model(dim_in, dim_out,
                                              dropout=cfg.gnn.dropout,
                                              residual=cfg.gnn.residual).model)
        else:
            self.post_mp = GNNHead(dim_in=dim_in, dim_out=dim_out)

        for _ in range(layers_mp):
            self.gnn_layers.append(conv_model(dim_in,
                                              dim_in,
                                              dropout=cfg.gnn.dropout,
                                              residual=cfg.gnn.residual))

    def build_conv_model(self, model_type):
        if model_type == 'gatedgcnconv':
            return GatedGCNLayer
        elif model_type == 'gineconv':
            return GINEConvLayer
        elif model_type == 'gcnconv':
            return GCNConvLayer
        elif model_type == 'gensconv':
            return GENConvLayer
        else:
            raise ValueError(f"Model {model_type} unavailable")


    def forward(self, batch):

        batch = self.encoder(batch)
        batch = self.pre_mp(batch)

        x0 = batch.x
        for i, conv in enumerate(self.gnn_layers):
            if cfg.gnn.layer_type == 'gensconv':
                batch = self.gnn_layers[i](batch, x0)
            else:
                batch = self.gnn_layers[i](batch)

        batch = self.post_mp(batch)
        return batch
