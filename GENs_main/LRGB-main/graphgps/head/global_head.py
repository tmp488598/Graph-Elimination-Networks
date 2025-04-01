import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head
from torch_geometric.graphgym.models.layer import new_layer_config, MLP


@register_head('global_san')
class SANGraphHead(nn.Module):
    """
    Transformer-based SAN prediction head for graph prediction tasks,
    with a top-K sparse attention mechanism on node embeddings.

    Args:
        dim_in (int):  Input node dimension.
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        L (int):       Number of hidden FC layers after pooling.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        # 1) 定义图级池化函数
        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]

        # 2) 定义 Transformer 超参数（可从 cfg 中读取，或手动写死）
        self.d_model = dim_in
        self.nhead = cfg.gt.attn_head if hasattr(cfg.gt, 'attn_head') else 4
        self.num_layers = cfg.gt.layers if hasattr(cfg.gt, 'layers') else 1
        self.dim_feedforward = cfg.gt.dim_ff if hasattr(cfg.gt, 'dim_ff') else 80
        dropout = cfg.gt.dropout if hasattr(cfg.gt, 'dropout') else 0.1
        self.activation = cfg.gt.activation if hasattr(cfg.gt, 'activation') else 'relu'
        self.k = cfg.gt.top_k if hasattr(cfg.gt, 'top_k') else 100  # 每个节点选 top-K

        # 3) 构造标准的 TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=dropout,
            activation=self.activation,
            batch_first=False  # 默认 (S, N, E)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

        dropout = cfg.gnn.dropout
        L = cfg.gnn.layers_post_mp

        layers = []
        for _ in range(L - 1):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(dim_in, dim_in, bias=True))
            layers.append(register.act_dict[cfg.gnn.act]())

        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dim_in, dim_out, bias=True))
        self.mlp = nn.Sequential(*layers)



    def _apply_index(self, batch):
        """
        将最终的图预测结果写入 batch.graph_feature，并返回 (pred, label)
        """
        return batch.graph_feature, batch.y

    @torch.no_grad()
    def build_topk_indices(self, x):
        """
        无梯度地计算全局相似度并选取每个节点的 top-K 邻居。

        Args:
            x (Tensor): shape = [N, d_model]

        Returns:
            topk_indices (LongTensor): shape = [N, k], 第 i 行是节点 i 的 top-K 邻居
        """

        # 1) 点积相似度
        sim = x @ x.t()  # (N, N)
        sim.fill_diagonal_(-1e10)  # 避免自己与自己
        # 2) top-K
        _, topk_indices = torch.topk(sim, k=self.k, dim=-1)
        return topk_indices

    def create_attn_mask(self, topk_indices, N):
        """
        基于 topk_indices 构造稀疏注意力的 mask 矩阵 attn_mask (N, N)。
        对 top-K 的位置填 0，其它位置填 -inf，让 Transformer 仅在 top-K 范围做注意力。

        Args:
            topk_indices (LongTensor): shape=[N, k]
            N (int): 节点总数

        Returns:
            attn_mask (Tensor): shape=[N, N], float32
        """
        device = topk_indices.device
        attn_mask = torch.full((N, N), float('-inf'), device=device)
        for i in range(N):
            attn_mask[i, topk_indices[i]] = 0.0
            # 如果想保留自连接，可加上:
            attn_mask[i, i] = 0.0
        return attn_mask

    def forward(self, batch):
        """
        batch.x: shape=(N, d_in), 节点特征
        batch.batch: shape=(N,), 记录节点所属图的索引 (若是多图场景)

        Returns:
            pred, label
        """
        # ========== 1) 基于节点特征，用 top-K 稀疏 Transformer 更新节点表示 ==========
        x = batch.x  # [N, d_in], d_in==self.d_model

        # 1.1) 计算 top-K 邻接 (无梯度)
        topk_indices = self.build_topk_indices(x)  # [N, k]

        # 1.2) 构造注意力 mask
        attn_mask = self.create_attn_mask(topk_indices, x.size(0))  # [N, N]

        # 1.3) 形状转换 => (S, N, E)，这里 S=N(序列长度), batch_size=1
        src = x.unsqueeze(1)  # => [N, 1, d_in]

        # 1.4) 送入 TransformerEncoder
        out = self.transformer_encoder(src, mask=attn_mask)  # [N, 1, d_in]
        out = out.squeeze(1)  # => [N, d_in]

        # 更新 batch.x (如果后续还需要用)
        batch.x = out

        # ========== 2) 做图级池化，得到图向量 graph_emb ==========
        if batch.x.size(0) !=  batch.y.size(0):
            out = self.pooling_fun(batch.x, batch.batch)  # => [num_graphs, d_in]

        # ========== 3) 多层线性映射 (与 SAN 模板一致) ==========

        # ========== 4) 写入 batch 并输出 ==========
        batch.graph_feature = self.mlp(out)
        pred, label = self._apply_index(batch)
        return pred, label
