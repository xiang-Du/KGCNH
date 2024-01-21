import dgl
import torch
from sklearn import metrics
import numpy as np
import pickle
from scipy.sparse import coo_matrix


def complex_score(head, relation, tail):
    """

    :param head: [batch_size, dim]
    :param relation: [batch_size, dim]
    :param tail: [batch_size, dim]
    :return: [batch_size]
    """
    re_head, im_head = torch.chunk(head, 2, dim=1)
    re_relation, im_relation = torch.chunk(relation, 2, dim=1)
    re_tail, im_tail = torch.chunk(tail, 2, dim=1)
    re_score = re_head * re_relation - im_head * im_relation
    im_score = re_head * im_relation + im_head * re_relation
    score = re_score * re_tail + im_score * im_tail
    score = score.sum(dim=1)
    return score


def calc_auc(labels, scores):
    fpr, tpr, _ = metrics.roc_curve(labels, scores)
    auc = metrics.auc(fpr, tpr)
    return auc


def calc_aupr(labels, scores):
    precision, recall, _ = metrics.precision_recall_curve(labels, scores)
    aupr = metrics.auc(recall, precision)
    return aupr


def save_model(result, path, mode='ab'):
    if mode == 'ab':
        with open(path, 'ab') as f:
            pickle.dump(result, f)
    else:
        with open(path, 'wb') as f:
            pickle.dump(result, f)


def load_model(path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
    return res


def print_config(config, args):
    print('data_path:', args.data_path)
    print('learning rate:', args.lr)
    print('embed_size:', args.embed_dim)
    print('weight_decay:', args.decay)
    print('dropout:', args.dropout)
    print('neg_ratio:', args.neg_ratio)
    print('enable_augmentation:', args.enable_augmentation)
    print('enable_gumbel:', args.enable_gumbel)
    print('gumbel tau:', args.tau)
    print('gumbel amplitude:', args.amplitude)
    print('valid_step', args.valid_step)

    for k, v in config.items():
        string = '{}:{}'.format(k, v)
        print(string)


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def exchange_head_tail(triple):
    h, r, t = triple
    return t, r, h


def generate_directed_coo_matrix(triples, dd_num):
    row = []
    data = []
    col = []
    dd_num = dd_num - 1
    for triple in triples:
        h, r, t = triple
        if h <= dd_num and t <= dd_num:
            row.append(h)
            data.append(r)
            col.append(t)
            row.append(t)
            data.append(r)
            col.append(h)
        elif h > dd_num and t > dd_num:
            print('check dataset')
        elif h <= dd_num < t:
            h, r, t = exchange_head_tail(triple)
            row.append(h)
            col.append(t)
            data.append(r)
        elif h > dd_num >= t:
            row.append(h)
            col.append(t)
            data.append(r)
    return (row, col), data


def exclude_isolation_point(train_set, valid_set):
    head_set = set()
    tail_set = set()
    exclusion_list = []
    valid = []
    valid_set = valid + valid_set
    for train_triple in train_set:
        h, _, t = train_triple
        head_set.add(h)
        tail_set.add(t)
    for valid_triple in valid_set:
        h, _, t = valid_triple
        if h not in head_set and t not in tail_set:
            exclusion_list.append(valid_triple)
            valid_set.remove(valid_triple)
    return valid_set, exclusion_list


def generate_bi_coo_matrix(kg_triple):
    row = []
    data = []
    col = []

    def bi_direct(triplet):
        h, r, t = triplet
        row.append(h)
        data.append(r)
        col.append(t)
        row.append(t)
        data.append(r)
        col.append(h)

    for triple in kg_triple:
        bi_direct(triple)
    return (row, col), data


def get_renumber_data(triple, drug_num):
    row = []
    rating = []
    col = []

    def direct(triplet):
        h, _, t = triplet
        row.append(h)
        rating.append(1)
        # renumber the disease id from 0 to disease_num -1
        col.append(t - drug_num)

    for triple in triple:
        direct(triple)
    return row, col, rating


def generate_enc_graph(triple, drug_num, disease_num):
    num_nodes_dict = {'drug': drug_num, 'disease': disease_num}
    row, col, rating = get_renumber_data(triple, drug_num)
    coo = coo_matrix((rating, (row, col)), shape=(drug_num, disease_num))
    adj = coo.todense()

    r, c = np.where(adj == 0)
    n_rating = []
    n_rating.extend([1] * len(r))
    # n_coo = coo_matrix((n_rating, (r, c)), shape=(drug_num, disease_num))
    data_dict = dict()
    data_dict.update({
        ('drug', str(0), 'disease'): (r, c),
        ('disease', 'rev-%s' % str(0), 'drug'): (c, r)
    })
    data_dict.update({
        ('drug', str(1), 'disease'): (row, col),
        ('disease', 'rev-%s' % str(1), 'drug'): (col, row)
    })
    graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

    return graph


def sample_gumbel(shape, device, lower_limit=1e-20, ):
    """Sample from Gumbel(0, 1)"""
    eps = torch.rand(shape).to(device)
    return -torch.log(-torch.log(eps + lower_limit) + lower_limit)


def gumbel_soft(weight, tau):
    """
    :param weight:  [n, x, 1]
    :param tau: temperature
    :return: [n, x, 1]
    """
    shape = weight.shape
    device = weight.device
    prob = weight
    neg_prob = 1 - prob
    prob = torch.cat((prob, neg_prob), dim=2)
    prob = prob + 1e-20
    eps = sample_gumbel(prob.shape, device)
    probability = torch.log(prob) + eps
    probability = probability / tau
    probability = torch.softmax(probability, dim=2)
    probability = probability[:, :, 0]
    return probability.view(shape)


def gumbel_scale(prob, gumbel_weight):
    scale = 1 / torch.sum(prob * gumbel_weight, dim=1, keepdim=True)
    return scale


def calc_norm(x):
    x = x.to('cpu').numpy().astype('float32')
    x[x == 0.] = np.inf
    x = torch.FloatTensor(1. / np.sqrt(x))
    return x.unsqueeze(1)


def comp(list1, list2):
    for val in list1:
        if val in list2:
            return True
    return False
