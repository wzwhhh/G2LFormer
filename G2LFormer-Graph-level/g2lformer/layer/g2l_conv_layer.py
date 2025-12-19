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
from torch_geometric.nn import Linear

import torch_geometric as pyg
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter, scatter_max, scatter_add
from g2lformer.utils import negate_edge_index
from torch_geometric.graphgym.register import *
import opt_einsum as oe

from torch import Tensor
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.nn.conv import MessagePassing
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg

from yacs.config import CfgNode as CN

import warnings

class LinearBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, activation: str = None, norm: str = None, negative_slope: float = 0.01):
        """
        :param activation: str, "relu", "leaky_relu", "tanh", "none" or None
        :param norm: str, "batchnorm", "none" or None
        :param negative_slope: float, negative_slope of leaky_relu, if activation != "leaky_relu", this parameter is invalid
        """
        super().__init__()
        # self.lin = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.lin = Linear(in_channels=in_features, out_channels=out_features, bias=bias)
        self.activation = lambda x: x
        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = F.tanh
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
            self.negative_slope = negative_slope
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "none" or activation is None:
            pass
        else:
            raise NotImplementedError("Activation function can only be ('relu', 'tanh', 'none', None)")

        self.norm = lambda x: x
        if norm == "batchnorm":
            self.norm = nn.BatchNorm1d(out_features)
        elif norm == "none" or norm is None:
            pass
        else:
            raise NotImplementedError("normalization can only be ('batchnorm', 'none', None)")

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if isinstance(self.norm, nn.BatchNorm1d):
            self.norm.reset_parameters()

    def forward(self, x):
        x = self.lin(x)
        x = self.norm(x)
        if self.activation == F.leaky_relu:
            x = self.activation(x, negative_slope=self.negative_slope)
        else:
            x = self.activation(x)
        return x

class NodeWeightLearner(nn.Module):
    def __init__(self, in_features: int, hid_features: int, numlayer: int = 2, negative_slope: float = 0.2, act = 'relu'):
        """
        :param in_features: input dim of extractor
        :param hid_features: output dim of extractor
        :param num_layer: num layer of learner
        :param negative_slope: negative slope of LeakyRelu
        """
        super().__init__()
        if numlayer < 1:
            raise ValueError("numlayer of Node Weight Learner cannot be less than 1")
        self.numlayer = numlayer
        self.extractor_1 = Linear(in_features, hid_features, bias=False)
        self.extractor_2 = Linear(in_features, hid_features, bias=False)
        learner_in_features = hid_features * 2
        self.learner = nn.ModuleList()
        for i in range(numlayer-1):
            self.learner.append(
                LinearBlock(learner_in_features,
                            learner_in_features,
                            bias=True,
                            activation=act,
                            norm="batchnorm",
                            negative_slope=negative_slope)
            )
        self.learner.append(
            LinearBlock(learner_in_features, 1, bias=False, activation=None, norm=None)
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.extractor_1.reset_parameters()
        self.extractor_2.reset_parameters()
        for i in range(self.numlayer):
            self.learner[i].reset_parameters()

    def forward(self, x1, x2=None):
        x1 = self.extractor_1(x1)
        if x2 is None:
            x2 = torch.zeros_like(x1)
        else:
            x2 = self.extractor_2(x2)
        learner_x = torch.cat([x1, x2], dim=1)
        for i in range(self.numlayer):
            learner_x = self.learner[i](learner_x)
        weight = torch.sigmoid(learner_x)
        return weight

class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_weight=True, use_init=False):
        super(GraphConvLayer, self).__init__()

        self.use_init = use_init
        self.use_weight = use_weight
        if self.use_init:
            in_channels_ = 2 * in_channels
        else:
            in_channels_ = in_channels
        self.W = nn.Linear(in_channels_, out_channels)

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x, edge_index, x0):
        N = x.shape[0]
        edge_weight = None
        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, N, False)
            row, col = edge_index
            # transposed if directed
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(N, N))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, N, False)
            edge_weight = None
            adj_t = edge_index
        x = matmul(adj_t, x)  # [N, D]
        # row, col = edge_index
        # d = degree(col, N).float()
        # d_norm_in = (1. / d[col]).sqrt()
        # d_norm_out = (1. / d[row]).sqrt()
        # value = torch.ones_like(row) * d_norm_in * d_norm_out
        # value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        # adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        # x = matmul(adj, x)  # [N, D]

        if self.use_init:
            x = torch.cat([x, x0], 1)
            x = self.W(x)
        elif self.use_weight:
            x = self.W(x)

        return x


class GraphConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.5, use_bn=True, use_residual=True,
                 use_weight=True, use_init=False, use_act=True):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers):
            self.convs.append(
                GraphConvLayer(hidden_channels, hidden_channels, use_weight, use_init))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index):
        layer_ = []

        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, layer_[0])
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x + layer_[-1]
        return x


class TransConvLayer(nn.Module):
    '''
    transformer with fast attention
    '''

    def __init__(self, in_channels,
                 out_channels,
                 num_heads,
                 use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, output_attn=False):
        # feature transformation
        qs = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        ks = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            vs = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
        else:
            vs = source_input.reshape(-1, 1, self.out_channels)

        # normalize input
        qs = qs / torch.norm(qs, p=2)  # [N, H, M]
        ks = ks / torch.norm(ks, p=2)  # [L, H, M]
        N = qs.shape[0]

        # numerator
        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
        attention_num += N * vs

        # denominator
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
        attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(
            attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        attn_output = attention_num / attention_normalizer  # [N, H, D]


        # compute attention for visualization if needed
        if output_attn:
            attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)  # [N, N]
            normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)  # [N,1]
            attention = attention / normalizer

        final_output = attn_output.mean(dim=1)

        if output_attn:
            return final_output, attention
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, num_heads=1,
                 dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=True, act='relu',ffn=False):
        super().__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        # self.activation = F.relu
        # self.activation = register.act_dict[act]
        if act == "relu":
            self.activation = F.relu
        elif act == "tanh":
            self.activation = F.tanh
        elif act == "leaky_relu":
            self.activation = F.leaky_relu
            self.negative_slope = negative_slope
        elif act == "gelu":
            self.activation = F.gelu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act
        self.ffn = True

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x):
        layer_ = []

        if self.ffn:
            # input MLP layer
            x = self.fcs[0](x)
            if self.use_bn:
                x = self.bns[0](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # store as residual link
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with full attention aggregation
            x = conv(x, x)
            if self.use_residual:
                x = (x + layer_[i]) / 2.
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.use_residual:
                x = (x + layer_[i]) / 2.
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]


class GatedGCNLayer(MessagePassing):
    """
        GatedGCN layer
        Residual Gated Graph ConvNets
        https://arxiv.org/pdf/1711.07553.pdf
    """

    def __init__(self, in_dim, out_dim, dropout, residual, ffn, act='relu',
                 equivstable_pe=False, **kwargs):
        super().__init__(**kwargs)
        self.activation = register.act_dict[act]
        self.A = nn.Linear(in_dim, out_dim, bias=True)
        self.B = nn.Linear(in_dim, out_dim, bias=True)
        self.C = nn.Linear(in_dim, out_dim, bias=True)
        self.D = nn.Linear(in_dim, out_dim, bias=True)
        self.E = nn.Linear(in_dim, out_dim, bias=True)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        self.EquivStablePE = equivstable_pe
        if self.EquivStablePE:
            self.mlp_r_ij = nn.Sequential(
                nn.Linear(1, out_dim),
                self.activation(),
                nn.Linear(out_dim, 1),
                nn.Sigmoid())

        self.act_fn_x = self.activation()
        self.act_fn_e = self.activation()
        self.dropout = dropout
        self.residual = residual
        self.e = None

        self.batch_norm = True
        self.ffn = ffn

        if self.batch_norm:
            self.bn_node_x = nn.BatchNorm1d(out_dim)
            self.bn_edge_e = nn.BatchNorm1d(out_dim)

        if self.ffn:
            # Feed Forward block.
            if self.batch_norm:
                self.norm1_local = nn.BatchNorm1d(out_dim)
            self.ff_linear1 = nn.Linear(out_dim, out_dim * 2)
            self.ff_linear2 = nn.Linear(out_dim * 2, out_dim)
            self.act_fn_ff = register.act_dict[cfg.gnn.act]()
            if self.batch_norm:
                self.norm2 = nn.BatchNorm1d(out_dim)
            self.ff_dropout1 = nn.Dropout(dropout)
            self.ff_dropout2 = nn.Dropout(dropout)

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        # return x
        return self.ff_dropout2(self.ff_linear2(x))

    def forward(self, batch):
        x, e, edge_index = batch.x, batch.edge_attr, batch.edge_index

        """
        x               : [n_nodes, in_dim]
        e               : [n_edges, in_dim]
        edge_index      : [2, n_edges]
        """
        if self.residual:
            x_in = x
            e_in = e

        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(e)
        Dx = self.D(x)
        Ex = self.E(x)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        pe_LapPE = batch.pe_EquivStableLapPE if self.EquivStablePE else None

        x, e = self.propagate(edge_index,
                              Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce,
                              e=e, Ax=Ax,
                              PE=pe_LapPE)
        if self.batch_norm:
            x = self.bn_node_x(x)
            e = self.bn_edge_e(e)

        x = self.act_fn_x(x)
        e = self.act_fn_e(e)

        x = F.dropout(x, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        if self.residual:
            x = x_in + x
            e = e_in + e

        batch.x = x
        batch.edge_attr = e

        if self.ffn:
            if self.batch_norm:
                batch.x = self.norm1_local(batch.x)

            batch.x = batch.x + self._ff_block(batch.x)

            if self.batch_norm:
                batch.x = self.norm2(batch.x)

        return batch

    def message(self, Dx_i, Ex_j, PE_i, PE_j, Ce):
        """
        {}x_i           : [n_edges, out_dim]
        {}x_j           : [n_edges, out_dim]
        {}e             : [n_edges, out_dim]
        """
        e_ij = Dx_i + Ex_j + Ce
        # e_ij = Dx_i + Ex_j
        sigma_ij = torch.sigmoid(e_ij)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        if self.EquivStablePE:
            r_ij = ((PE_i - PE_j) ** 2).sum(dim=-1, keepdim=True)
            r_ij = self.mlp_r_ij(r_ij)  # the MLP is 1 dim --> hidden_dim --> 1 dim
            sigma_ij = sigma_ij * r_ij

        self.e = e_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        """
        sigma_ij        : [n_edges, out_dim]  ; is the output from message() function
        index           : [n_edges]
        {}x_j           : [n_edges, out_dim]
        """
        dim_size = Bx.shape[0]  # or None ??   <--- Double check this

        sum_sigma_x = sigma_ij * Bx_j
        numerator_eta_xj = scatter(sum_sigma_x, index, 0, None, dim_size,
                                   reduce='sum')

        sum_sigma = sigma_ij
        denominator_eta_xj = scatter(sum_sigma, index, 0, None, dim_size,
                                     reduce='sum')

        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)
        return out

    def update(self, aggr_out, Ax):
        """
        aggr_out        : [n_nodes, out_dim] ; is the output from aggregate() function after the aggregation
        {}x             : [n_nodes, out_dim]
        """
        x = Ax + aggr_out
        e_out = self.e
        del self.e
        return x, e_out

class GNNConv(nn.Module):
    def __init__(self, dim_in, dim_out, dropout, residual, ffn, classic_gnn_num_layers = 1,nw_num_layers=1,act='relu'):
        super(GNNConv, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual
        self.batch_norm = True
        self.ffn = ffn
        self.act = register.act_dict[act]

        self.classic_gnn_num_layers = classic_gnn_num_layers
        self.learners = nn.ModuleList()
        self.learners.append(
            NodeWeightLearner(in_features=dim_in,
                              hid_features=dim_out,
                              numlayer=nw_num_layers,
                              negative_slope=0.1,
                              act=act)
        )  # input learner

        self.convs = torch.nn.ModuleList()

        self.convs.append(GatedGCNLayer(dim_in, dim_out, dropout, residual, ffn, act=act,
                 equivstable_pe=False))

        if self.batch_norm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(dim_in))

        if self.residual:
            self.lins = torch.nn.ModuleList()
            self.lins.append(torch.nn.Linear(dim_in, dim_out))

        for _ in range(classic_gnn_num_layers - 1):
            self.learners.append(
                NodeWeightLearner(in_features=dim_out,
                                  hid_features=dim_out,
                                  numlayer=nw_num_layers,
                                  negative_slope=0.1,
                                  act=act)
            )
            self.convs.append(GatedGCNLayer(dim_out, dim_out, dropout, residual, ffn, act=act,
                 equivstable_pe=False))
            if self.batch_norm:
                self.bns.append(torch.nn.BatchNorm1d(dim_out))
            if self.residual:
                self.lins.append(torch.nn.Linear(dim_out, dim_out))

        if self.ffn:
            # Feed Forward block.
            if self.batch_norm:
                self.norm1_local = nn.BatchNorm1d(dim_out)
            self.ff_linear1 = nn.Linear(dim_out, dim_out * 2)
            self.ff_linear2 = nn.Linear(dim_out * 2, dim_out)
            self.act_fn_ff = F.relu
            if self.batch_norm:
                self.norm2 = nn.BatchNorm1d(dim_out)
            self.ff_dropout1 = nn.Dropout(dropout)
            self.ff_dropout2 = nn.Dropout(dropout)

    def reset_parameters(self):
        # for learner in self.learners:
        #     learner.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def forward(self, batch):
        x = batch.x
        nw = self.learners[0](x)  # node weight
        x = x * nw
        fused_node_rep = x
        batch.x = x
        for i, conv in enumerate(self.convs[:-1]):
            # if self.residual:
            #     batch_x_in = self.lins[i](batch.x)
            batch = conv(batch)
            x = batch.x
            # if self.batch_norm:
            #     x = self.bns[i](x)
            nw = self.learners[i + 1](x, fused_node_rep)
            x = x * nw
            # x = self.act(x)
            # if self.residual:
            #     x = batch_x_in + x  # Residual connection.
            if self.ffn:
                if self.batch_norm:
                    x = self.norm1_local(x)
                x = x + self._ff_block(x)
                if self.batch_norm:
                    x = self.norm2(x)
            fused_node_rep = fused_node_rep + x
            batch.x = x
        return self.convs[-1](batch)

@register_layer("g2lformer")
class G2LFormer(nn.Module):
    def __init__(self, in_dim, out_dim,
                 trans_num_layers=1, trans_num_heads=1, attn_dropout=0.5, trans_use_bn=True,
                 trans_use_weight=True, trans_use_act=True,

                 use_classic_gnn=True, classic_gnn_num_layers=1, nw_num_layers = 1, classic_gnn_dropout=0.2,
                 dropout = 0.,
                 layer_norm=False, batch_norm=True,
                 residual=True,
                 act='relu',
                 ffn=False,
                 cfg=dict(),
                 need_learner = False,
                 ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.ffn = ffn
        self.act = register.act_dict[act]

        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.norm1_local = nn.LayerNorm(out_dim)
            self.norm1_attn = nn.LayerNorm(out_dim)
            self.norm2 = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(out_dim)
            self.norm1_attn = nn.BatchNorm1d(out_dim)
            self.norm2 = nn.BatchNorm1d(out_dim)

        trans_use_bn = self.batch_norm
        trans_dropout = attn_dropout
        self.attention = TransConv(in_dim, out_dim, trans_num_layers, trans_num_heads, trans_dropout,
                                   trans_use_bn, residual, trans_use_weight, trans_use_act, act=act,ffn=ffn)

        self.use_classic_gnn = use_classic_gnn
        if self.use_classic_gnn:
            self.gnn_conv2 = GNNConv(out_dim, out_dim, residual = residual, ffn = ffn, classic_gnn_num_layers = classic_gnn_num_layers,
                                     dropout = classic_gnn_dropout, nw_num_layers = nw_num_layers, act=act)

        self.need_learner = need_learner
        if self.need_learner:
            self.learners = nn.ModuleList()
            self.learners.append(
                NodeWeightLearner(in_features=out_dim,
                                  hid_features=out_dim,
                                  numlayer=nw_num_layers,
                                  negative_slope=0.1,
                                  act=act)
            )  # input learner

    def forward(self, batch):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        h = batch.x
        h_in1 = h  # for first residual connection
        h_attn = self.attention(h)
        if self.need_learner:
            nw = self.learners[0](h_attn, h_in1)
            h_attn = h_attn * nw
        #
        # h_attn = self.dropout_attn(h_attn)
        # if self.residual:
        #     h_attn = h_in1 + h_attn  # Residual connection.
        # if self.layer_norm:
        #     h_attn = self.norm1_attn(h_attn, batch.batch)
        # if self.batch_norm:
        #     h_attn = self.norm1_attn(h_attn)
        batch.x = h_attn
        batch = self.gnn_conv2(batch)

        return batch

    def get_attentions(self, x):
        attns = self.attention.get_attentions(x)  # [layer num, N, N]
        return attns

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        if self.use_classic_gnn:
            self.gnn_conv2.reset_parameters()