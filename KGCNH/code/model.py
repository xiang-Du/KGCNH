import torch
from torch import nn
from torch.nn import functional as fn
from layers import KGCNH
import numpy as np


class ABCModel(nn.Module):

    def __init__(self):
        super(ABCModel, self).__init__()

    @staticmethod
    def _l2_norm(x):
        return (x * x).sum()

    def l2_loss_mean(self, *args):
        loss = 0
        for x in args:
            loss += self._l2_norm(x)
        return loss

    def calc_reg_loss(self, pos_head, pos_tail, neg_tail, *args):
        reg_loss = ((pos_head * pos_head).sum() + (pos_tail * pos_tail).sum() + (neg_tail * neg_tail).sum())
        other_loss = 0
        if len(args) > 0:
            other_loss = self.l2_loss_mean(*args)
        reg_loss = reg_loss + other_loss
        batch_size = (2 * (pos_head.shape[0] + neg_tail.shape[0] * neg_tail.shape[1]))
        return reg_loss / batch_size

    @staticmethod
    def calc_bpr_loss(pos_score, neg_score):

        # utilize the broadcast mechanism
        score = pos_score - neg_score
        loss = -fn.logsigmoid(score).mean()
        return loss

    @staticmethod
    def calc_bce_loss(pos_score, neg_score):

        score = torch.cat((pos_score.contiguous().view(-1), neg_score.contiguous().view(-1)))
        pos_label = torch.ones_like(pos_score)
        neg_label = torch.zeros_like(neg_score.view(-1))
        label = torch.cat((pos_label, neg_label)).to(score)
        loss = fn.binary_cross_entropy_with_logits(score, label)
        return loss


class Model(ABCModel):
    def __init__(self, entity_num, drug_num, disease_num, relation_num, embed_dim,
                 model_args, enable_augmentation, gumbel_args, weight_decay, p_drop, score_fun='dot'):

        super(Model, self).__init__()
        self.entity_num = entity_num
        self.drug_num = drug_num
        self.relation_num = relation_num
        self.embed_dim = embed_dim
        self.drug_disease_num = drug_num + disease_num

        init = torch.zeros((entity_num, embed_dim))
        init_r = torch.zeros((relation_num, embed_dim))
        gain = nn.init.calculate_gain('relu')
        torch.nn.init.xavier_normal_(init, gain=gain)
        torch.nn.init.xavier_normal_(init_r, gain=gain)
        self.entity_embed = nn.Parameter(init)
        self.relation_embed = nn.Parameter(init_r)
        self.enable_augmentation = enable_augmentation
        self.gumbel_args = gumbel_args
        if enable_augmentation:
            index = [i for i in range(entity_num)]
            np.random.shuffle(index)
            init_augment = init[index]
            self.augment_entity_embed = nn.Parameter(init_augment)
            self.augment_relation_embed = nn.Parameter(init_r)

        self.entity_num = entity_num
        self.disease_num = disease_num
        self.relation_num = relation_num
        self.embed_dim = embed_dim

        self.layer_num = model_args[0]
        self.head_num = model_args[1]
        self.weight_decay = weight_decay
        self.p_drop = p_drop
        self.kgcnh_layer = nn.ModuleList()
        for _ in range(self.layer_num):
            self.kgcnh_layer.append(
                KGCNH((entity_num, embed_dim), self.head_num, p_drop, enable_augmentation, gumbel_args))
        self.score_fun = score_fun
        self.mlp = None
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if self.score_fun == 'mlp':
            self.mlp = nn.Sequential(nn.Linear(self.embed_dim * 2, self.embed_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.embed_dim, 1))
            for m in self.mlp:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=gain)

    def __get_kg_embed(self, g, relation):

        relation_embed = self.relation_embed[relation]
        embed = self.entity_embed
        for i in range(self.layer_num):
            embed = self.kgcnh_layer[i](g, embed, relation_embed)

        return embed

    def __get_augment_embed(self, g, relation):
        relation_embed = self.augment_relation_embed[relation]
        embed = self.augment_entity_embed

        for i in range(self.layer_num):
            embed = self.kgcnh_layer[i](g, embed, relation_embed)
        return embed

    @staticmethod
    def calc_dot_score(embed, h, t, n_t):

        h_embed = embed[h]
        t_embed = embed[t]
        # pos_score [n,1]
        pos_score = torch.multiply(h_embed, t_embed).sum(-1)
        n_t_embed = embed[n_t]
        neg_score = (h_embed * n_t_embed).sum(-1)
        return pos_score, neg_score

    def calc_mlp_score(self, embed, h, t, n_t):
        h_embed = embed[h]
        t_embed = embed[t]
        # concatenate two embed
        pos_embed = torch.multiply(h_embed, t_embed)
        pos_score = self.mlp(pos_embed)
        pos_score = pos_score.squeeze(dim=2)
        # change h[n] to [n, neg_ratio] =>[n*neg_ratio]
        h_ex = h.expand(n_t.shape)
        h_ex_embed = embed[h_ex]
        n_t_embed = embed[n_t]
        neg_embed = torch.multiply(h_ex_embed, n_t_embed)
        neg_score = self.mlp(neg_embed)
        neg_score = neg_score.squeeze(dim=2)
        return pos_score, neg_score

    def calc_contrastive_loss(self, kg_embed, augment_embed):
        return self.contrastive(kg_embed, augment_embed)

    def train_step(self, kg_graph, graph, relation, relation2, h, t, n_t):
        if self.enable_augmentation:

            # pos_score => [n, 1] neg_score =>[n, neg_ratio]
            pos_score, neg_score, embed, kg_entity_embed, augment_embed = self.__predict_with_augment(kg_graph,
                                                                                                      graph,
                                                                                                      relation,
                                                                                                      relation2, h,
                                                                                                      t,
                                                                                                      n_t)
            loss1 = self.calc_bpr_loss(pos_score, neg_score)

            loss = loss1
        else:
            pos_score, neg_score, embed = self.__predict_wo_augment(kg_graph, relation, h, t, n_t)
            loss = self.calc_bpr_loss(pos_score, neg_score)

        reg_loss = self.calc_reg_loss(embed[h], embed[t], embed[n_t], self.relation_embed)

        reg_loss = self.weight_decay * reg_loss
        return loss + reg_loss, reg_loss

    def __predict_with_augment(self, kg_graph, augment_graph, kg_relation, augment_relation, h, t, n_t):

        kg_entity_embed = self.__get_kg_embed(kg_graph, kg_relation)
        augment_embed = self.__get_augment_embed(augment_graph, augment_relation)
        if self.training:
            # embed = kg_entity_embed
            embed = torch.cat((kg_entity_embed, augment_embed), dim=1)
        else:
            embed = torch.cat((kg_entity_embed, augment_embed), dim=1)

        if self.score_fun == 'mlp':
            pos_score, neg_score = self.calc_mlp_score(embed, h, t, n_t)
        else:
            pos_score, neg_score = self.calc_dot_score(embed, h, t, n_t)
        return pos_score, neg_score, embed, kg_entity_embed, augment_embed

    def predict(self, kg_graph, graph, relation, g_relation, h, t, n_t):
        if self.enable_augmentation:
            pos_score, neg_score, embed, _, _ = self.__predict_with_augment(kg_graph, graph, relation, g_relation, h, t,
                                                                            n_t)
        else:
            pos_score, neg_score, embed = self.__predict_wo_augment(kg_graph, relation, h, t, n_t)
        score = torch.cat((pos_score.contiguous().view(-1), neg_score.contiguous().view(-1)))
        return score, embed.detach()

    def __predict_wo_augment(self, kg_graph, relation, h, t, n_t):

        kg_entity_embed = self.__get_kg_embed(kg_graph, relation)
        embed = kg_entity_embed
        if self.score_fun == 'mlp':
            pos_score, neg_score = self.calc_mlp_score(embed, h, t, n_t)
        else:
            pos_score, neg_score = self.calc_dot_score(embed, h, t, n_t)
        return pos_score, neg_score, embed.detach()
