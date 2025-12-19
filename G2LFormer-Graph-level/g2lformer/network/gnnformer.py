import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree, to_undirected, remove_self_loops, add_self_loops
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import GATConv, GCNConv, ClusterGCNConv, GCN2Conv, SAGEConv

from g2lformer.layer.g2l_conv_layer import TransConv, GraphConv, GNNConv

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.register import register_network
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)

from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP

# class FeatureEncoder(torch.nn.Module):
#     """
#     Encoding node and edge features
#
#     Args:
#         dim_in (int): Input feature dimension
#     """
#     def __init__(self, dim_in):
#         super(FeatureEncoder, self).__init__()
#         self.dim_in = dim_in
#         if cfg.dataset.node_encoder:
#             # Encode integer node features via nn.Embeddings
#             NodeEncoder = register.node_encoder_dict[
#                 cfg.dataset.node_encoder_name]
#             self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
#             if cfg.dataset.node_encoder_bn:
#                 self.node_encoder_bn = BatchNorm1dNode(
#                     new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
#                                      has_bias=False, cfg=cfg))
#             # Update dim_in to reflect the new dimension of the node features
#             self.dim_in = cfg.gnn.dim_inner
#         if cfg.dataset.edge_encoder:
#             # Hard-limit max edge dim for PNA.
#             if 'PNA' in cfg.gt.layer_type:
#                 cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
#             else:
#                 cfg.gnn.dim_edge = cfg.gnn.dim_inner
#             # Encode integer edge features via nn.Embeddings
#             EdgeEncoder = register.edge_encoder_dict[
#                 cfg.dataset.edge_encoder_name]
#             self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
#             if cfg.dataset.edge_encoder_bn:
#                 self.edge_encoder_bn = BatchNorm1dNode(
#                     new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
#                                      has_bias=False, cfg=cfg))
#
#     def forward(self, batch):
#         for module in self.children():
#             batch = module(batch)
#         return batch

@register_network('gnnformer')
class GNNFormer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        global_model_type = cfg.gt.get('layer_type', "g2lformer")
        print("layer type is:", global_model_type)

        TransformerLayer = register.layer_dict.get(global_model_type)

        layers = []
        for i in range(cfg.gt.layers):
            layers.append(TransformerLayer(
                in_dim=dim_in,
                out_dim=dim_in,
                trans_num_heads=cfg.gt.n_heads,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                act=cfg.gt.act,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                residual=cfg.gt.residual,
                use_classic_gnn=cfg.gt.use_classic_gnn,
                classic_gnn_num_layers=cfg.gt.classic_gnn_num_layers,
                nw_num_layers=cfg.gt.nw_num_layers,
                classic_gnn_dropout=cfg.gt.classic_gnn_dropout,
                ffn=cfg.gt.ffn,
                need_learner = False if i == 0 else True
            ))

        self.layers = torch.nn.Sequential(*layers)

        # self.gnn_layers = torch.nn.Sequential(*layers)
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch