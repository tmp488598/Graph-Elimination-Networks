import torch
from torch import nn
import torch_geometric.nn as gnn
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from graphgps.layer.neuralwalker_layer import NeuralWalkerLayer

@register_network('NeuralWalker')
class NeuralWalker(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.walk_encoder = cfg.gnn.walker.walk_encoder
        global_mp_type = cfg.gnn.walker.global_mp_type

        self.encoder = FeatureEncoder(dim_in)
        hidden_size = self.encoder.dim_in

        # Walk encoder

        self.blocks = nn.ModuleList([
            NeuralWalkerLayer(
                hidden_size=hidden_size,
                sequence_layer_type=self.walk_encoder,
                d_state=cfg.gnn.walker.d_state,
                d_conv=cfg.gnn.walker.d_conv,
                expand=cfg.gnn.walker.expand,
                mlp_ratio=cfg.gnn.walker.mlp_ratio,
                use_encoder_norm=cfg.gnn.walker.use_encoder_norm,
                proj_mlp_ratio=cfg.gnn.walker.proj_mlp_ratio,
                walk_length=cfg.gnn.walker.walk_length,
                use_positional_encoding=cfg.gnn.walker.use_positional_encoding,
                pos_embed=cfg.gnn.walker.walk_pos_embed,
                window_size=cfg.gnn.walker.window_size,
                bidirection=cfg.gnn.walker.bidirection,
                layer_idx=i,
                local_gnn_type=cfg.gnn.walker.local_mp_type,
                global_model_type=None if global_mp_type == 'vn' and i == cfg.gnn.layers_mp - 1 else global_mp_type,
                num_heads=cfg.gnn.walker.num_heads,
                dropout=cfg.gnn.dropout,
                attn_dropout=cfg.gnn.walker.attn_dropout,
                vn_norm_first=cfg.gnn.walker.vn_norm_first,
                vn_norm_type=cfg.gnn.walker.vn_norm_type,
                vn_pooling=cfg.gnn.walker.vn_pooling,
            ) for i in range(cfg.gnn.layers_mp)
        ])

        self.node_out = None
        if cfg.gnn.walker.node_out:
            if global_mp_type is None or global_mp_type == 'vn':
                self.node_out = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(cfg.gnn.dropout),
                    nn.Linear(hidden_size, hidden_size)
                )

        GNNHead = register.head_dict[cfg.gnn.head]

        self.out_head = GNNHead(dim_in=hidden_size, dim_out=dim_out)

    def forward(self, batch):
        batch.walk_pe = torch.cat(
            [batch.walk_node_id_encoding, batch.walk_node_adj_encoding], dim=-1
        )

        batch = self.encoder(batch)

        for i, block in enumerate(self.blocks):
            batch = block(batch)

        h = batch.x
        if self.node_out is not None:
            h = self.node_out(h)

        batch.x = h

        return self.out_head(batch)

    def get_params(self):
        if self.walk_encoder == "s4":
            # All parameters in the model
            all_parameters = list(self.parameters())

            # General parameters don't contain the special _optim key
            param_groups = [{"params": [p for p in all_parameters if not hasattr(p, "_optim")]}]

            # Add parameters with special hyperparameters
            hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
            hps = [
                dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
            ]  # Unique dicts
            for hp in hps:
                params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
                param_groups.append(
                    {"params": params, **hp}
                )

            # Print optimizer info
            keys = sorted(set([k for hp in hps for k in hp.keys()]))
            for i, g in enumerate(param_groups):
                group_hps = {k: g.get(k, None) for k in keys}
                print(' | '.join([
                    f"Optimizer group {i}",
                    f"{len(g['params'])} tensors",
                ] + [f"{k} {v}" for k, v in group_hps.items()]))

            return param_groups
        return self.parameters()
