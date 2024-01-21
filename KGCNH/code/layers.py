import torch
import torch.nn.functional as fn
from utils import gumbel_soft

import torch.nn as nn


class Normal(nn.Module):
    def __init__(self, feature_shape, eps=1e-10):
        super(Normal, self).__init__()
        self.gamma = nn.Parameter(torch.ones(feature_shape))
        self.bias = nn.Parameter(torch.zeros(feature_shape))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / torch.pow((std + self.eps), 0.5) + self.bias


class KGCNHLayer(nn.Module):
    def __init__(self, in_dim, p_drop, enable_augmentation, gumbel_args):
        super(KGCNHLayer, self).__init__()
        self.enable_augmentation = enable_augmentation
        self.enable_gumbel, tau, self.step = gumbel_args
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(in_dim, in_dim, bias=False)
        self.attn_fc = nn.Linear(in_dim * 2, 1, bias=False)
        self.reset_parameters()
        self.tau = nn.Parameter(torch.tensor(tau), requires_grad=False)
        self.annealing = 0

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):

        d = edges.dst["h"]
        dr = torch.multiply(d, edges.data['r'])
        h = edges.src['h']
        dr = self.fc(dr)
        h = self.fc(h)
        z = torch.cat((h, dr), dim=1)
        a = self.attn_fc(z)
        return {"e": fn.leaky_relu(a) / 0.05}

    def message_func(self, edges):
        if self.training:
            if self.enable_augmentation:
                if self.annealing % 2 == 0:
                    self.tau = nn.Parameter(self.tau - self.step, requires_grad=False)
            else:
                self.tau = nn.Parameter(self.tau - self.step, requires_grad=False)
            self.annealing = self.annealing + 1
        return {"z": edges.src["h"], "e": edges.data["e"]}

    def reduce_func(self, nodes):
        tau = self.tau
        alpha = fn.softmax(nodes.mailbox["e"], dim=1)
        if self.enable_gumbel:
            gumbel_weight = gumbel_soft(alpha, tau)
            h = torch.sum(gumbel_weight * nodes.mailbox["z"], dim=1)
        else:
            h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
        return {"h": h}

    def forward(self, g, h, r):
        h = self.drop(h)
        r = self.drop(r)
        with g.local_scope():
            g.ndata['h'] = h
            g.edata['r'] = r
            g.apply_edges(self.edge_attention)
            g.update_all(self.message_func, self.reduce_func)
            feat = g.ndata.pop("h")
            return feat


class MultiHeadKGCNHLayer(nn.Module):
    def __init__(self, in_dim, num_heads, p_drop, enable_contrastive, gumbel_args):
        super(MultiHeadKGCNHLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(KGCNHLayer(in_dim, p_drop, enable_contrastive, gumbel_args))
        self.enable_contrastive = enable_contrastive

    def forward(self, g, h, r):
        feat = [attn_head(g, h, r) for attn_head in self.heads]
        return torch.mean(torch.stack(feat, dim=1), dim=1)


class KGCNH(nn.Module):
    def __init__(self, feature_shape, num_heads, p_drop, enable_contrastive, gumbel_args):
        super(KGCNH, self).__init__()
        self.feature_shape = feature_shape
        self.enable_contrastive = enable_contrastive

        self.p_drop = p_drop
        self.mkgcnh = MultiHeadKGCNHLayer(feature_shape[-1], num_heads, p_drop, enable_contrastive, gumbel_args)
        self.drop = nn.Dropout(p_drop)
        self.normal = Normal(feature_shape)

    def forward(self, g, feat, r):
        node_num = g.num_nodes()
        feat_x = feat[:node_num]

        conv_feat = self.mkgcnh(g, feat_x, r)
        res_feat = feat_x + conv_feat
        res_feat = self.drop(self.normal(res_feat))
        return torch.cat((res_feat, feat[node_num:]))
