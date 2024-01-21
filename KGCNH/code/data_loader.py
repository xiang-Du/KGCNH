import copy
import random
import torch
import numpy as np
from collections import defaultdict
import os
import pickle
from torch.utils.data import Dataset
from tqdm import tqdm
import shutil


class DataProcessor(object):
    def __init__(self, data_path='./data/DrugBank', hop=None) -> None:
        if hop is not None:
            self.root = os.path.join(data_path, str(hop))
            if not os.path.exists(self.root):
                os.makedirs(self.root)
                shutil.copy(os.path.join(data_path, 'triple.txt'), os.path.join(self.root, 'triple.txt'))
        else:
            self.root = data_path
        self.hop = hop
        self.sample_subset = None
        self.entities2id, self.relations2id = self._get_entities_and_relations_id()
        self._drug_disease_num = None
        self._drug_num = None
        self._disease_num = None 

    @property
    def drug_disease_num(self):
        if self._drug_disease_num is None:
            self._calc_drug_disease_num()
        return self._drug_disease_num

    @property
    def disease_num(self):
        if self._disease_num is None:
            self._calc_drug_disease_num()
        return self._disease_num

    @property
    def drug_num(self):
        if self._drug_num is None:
            self._calc_drug_disease_num()
        return self._drug_num

    def get_relation_num(self):
        return len(self.relations2id)

    def get_node_num(self):
        return len(self.entities2id)

    def get_entities2id(self):
        return self.entities2id

    def get_relations2id(self):
        return self.relations2id

    def triple2id(self, triple):

        h, r, t = triple
        h = self.entities2id[h]
        r = self.relations2id[r]
        t = self.entities2id[t]
        return h, r, t

    def _calc_drug_disease_num(self):
        res = self._load_data('treats')
        drug_num = -1
        max_id = -1
        for triple in res:
            h, r, t = self.triple2id(triple)
            drug_num = max(drug_num, h)
            max_id = max(max_id, h, t)
        self._drug_disease_num = max_id + 1
        self._drug_num = drug_num + 1
        self._disease_num = max_id - drug_num

    def load_data(self, mode='treats'):
        if mode == 'treats':
            path = os.path.join(self.root, 'drug_treats_disease.pkl')
        else:
            path = os.path.join(self.root, 'other.pkl')
        with open(path, 'rb') as f:
            res = pickle.load(f)
        triples = []
        for i in res:
            triple_id = self.triple2id(i)
            triples.append(triple_id)
        return triples

    def _load_data(self, name='treats'):

        path1 = os.path.join(self.root, 'drug_treats_disease.pkl')
        path2 = os.path.join(self.root, 'other.pkl')
        if not os.path.exists(path1) or not os.path.exists(path2):
            self._extract_data(path1, path2)
        if name == 'treats':
            with open(path1, 'rb') as f:
                res = pickle.load(f)
        else:
            with open(path2, 'rb') as f:
                res = pickle.load(f)
        return res

    def _extract_data(self, path1, path2):
        drug_disease_triple = []
        kg_triple = []
        drug_disease_node = set()
        with open(os.path.join(self.root, 'triple.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                h, r, t = line.strip().split('\t')
                if r == 'treats':
                    drug_disease_triple.append((h, r, t))
                    drug_disease_node.add(h)
                    drug_disease_node.add(t)
                else:
                    kg_triple.append((h, r, t))
        if self.hop is not None:
            def sample_subset(nodes):
                result = []
                for i, l in enumerate(nodes):
                    hop_neighbor = set()
                    hop_triple = []
                    for triple in tqdm(kg_triple):
                        h1, _, t1 = triple
                        if h1 in l:
                            hop_neighbor.add(t1)
                            hop_triple.append(triple)
                        elif t1 in l:
                            hop_neighbor.add(h1)
                            hop_triple.append(triple)
                    nodes.append(list(hop_neighbor))
                    result += hop_triple
                    if i == self.hop - 1:
                        break
                return list(set(result))

            res = sample_subset([drug_disease_node])

        else:
            res = kg_triple
        with open(path1, 'wb') as f:
            pickle.dump(drug_disease_triple, f)
        with open(path2, 'wb') as f:
            pickle.dump(res, f)

    def _get_entities_and_relations_id(self):
        path1 = os.path.join(self.root, 'entities.pkl')
        path2 = os.path.join(self.root, 'relations.pkl')
        if not os.path.exists(path1) or not os.path.exists(path2):
            self._create_entities_and_relations_id(path1, path2)
        with open(path1, 'rb') as f:
            entities2id = pickle.load(f)
        with open(path2, 'rb') as f:
            relations2id = pickle.load(f)
        return entities2id, relations2id

    def _create_entities_and_relations_id(self, path1, path2):
        relations_id = 0
        relations2id = {}
        entities_id = 0
        entities2id = {}
        treats_data = self._load_data('treats')
        other_data = self._load_data('other')

        def add2entity_dict(entity):
            nonlocal entities_id, entities2id
            if entity not in entities2id.keys():
                entities2id[entity] = entities_id
                entities_id = entities_id + 1

        def add2relation_dict(relation):
            nonlocal relations_id, relations2id
            if relation not in relations2id.keys():
                relations2id[relation] = relations_id
                relations_id = relations_id + 1

        # load drug and disease first
        for data in treats_data:
            h, _, _ = data
            add2entity_dict(h)
        for data in treats_data:
            _, r, t = data
            add2entity_dict(t)
            add2relation_dict(r)
        for data in other_data:
            h, r, t = data
            add2entity_dict(h)
            add2entity_dict(t)
            add2relation_dict(r)

        with open(path1, 'wb') as f:
            pickle.dump(entities2id, f)
        with open(path2, 'wb') as f:
            pickle.dump(relations2id, f)


class KGDataset(Dataset):
    def __init__(self, kg_triple, neg_ratio):
        super(KGDataset, self).__init__()

        self.kg_triple = kg_triple
        self.dataset_num = len(kg_triple)
        self.neg_ratio = neg_ratio
        self.num_entities = 0
        self.tail = None
        self.head = None
        self.head_r_dict = None
        self.r_tail_dict = None
        self.process_all_tripe()

    def process_all_tripe(self):
        num_entities = 0
        head = set()
        tail = set()
        head_r_dict = defaultdict(list)
        r_tail_dict = defaultdict(list)
        for triple in self.kg_triple:
            h, r, t = triple
            num_entities = max(num_entities, h, t)
            head.add(h)
            tail.add(t)
            head_r_dict[(h, r)].append(t)
            r_tail_dict[(r, t)].append(h)
        self.head_r_dict = head_r_dict
        self.r_tail_dict = r_tail_dict
        self.head = np.array(list(head))
        self.tail = np.array(list(tail))

    def __neg_sample(self, x, y, num, mode='head'):
        neg_samples = []
        n = 0
        if mode == 'head':
            sample_data = self.head
            pos_data = self.r_tail_dict

        else:
            sample_data = self.tail
            pos_data = self.head_r_dict

        while n < num:
            negative_sample = np.random.randint(
                len(sample_data), size=min(len(sample_data), self.neg_ratio * 2))
            mask = np.in1d(
                sample_data[negative_sample],
                pos_data[(x, y)],
                assume_unique=True,
                invert=True
            )
            negative_sample = sample_data[negative_sample[mask]]
            # negative_sample = np.random.choice(
            #     sample_data, size=min(len(sample_data), self.neg_ratio * 2), replace=False)
            # mask = np.in1d(
            #     negative_sample,
            #     pos_data[(x, y)],
            #     assume_unique=True,
            #     invert=True
            # )
            # negative_sample = negative_sample[mask]
            n = n + len(neg_samples)
            neg_samples.append(negative_sample)
        neg_samples = np.concatenate(neg_samples)[:num]
        return neg_samples

    def generate_neg_sample(self, h, r, t):
        head_neg = self.neg_ratio // 2
        tail_neg = self.neg_ratio - head_neg
        head_neg_samples = self.__neg_sample(r, t, head_neg, 'head')
        tail_neg_samples = self.__neg_sample(h, r, tail_neg, 'tail')
        return head_neg_samples, tail_neg_samples

    def __len__(self):
        return self.dataset_num

    def __getitem__(self, item):
        positive_sample = self.kg_triple[item]
        h, r, t = positive_sample
        head_neg_samples, tail_neg_samples = self.generate_neg_sample(h, r, t)
        positive_label = [1]
        negative_label = [0 for _ in range(self.neg_ratio)]
        return torch.tensor(positive_sample), torch.tensor(head_neg_samples), torch.tensor(
            tail_neg_samples), torch.tensor(positive_label, dtype=torch.float), torch.tensor(negative_label,
                                                                                             dtype=torch.float)


class DrugDiseaseDataSet(Dataset):
    def __init__(self, sub_triple, all_triple, neg_ratio) -> None:
        super(DrugDiseaseDataSet, self).__init__()
        self.sub_triple = sub_triple
        self.all_triple = all_triple
        self.neg_ratio = neg_ratio
        self.num_entities = 0
        self.tail = None
        self.len_tail = 0
        self.process_all_tripe()

    def process_all_tripe(self):
        num_entities = 0
        triple_dict = defaultdict(list)
        tail = set()
        for triple in self.all_triple:
            h, r, t = triple
            num_entities = max(num_entities, h, t)
            tail.add(t)

            triple_dict[(h, r)].append(t)
        self.all_triple = triple_dict
        self.num_entities = num_entities + 1
        self.tail = np.array(list(tail))
        self.len_tail = len(self.tail)

    def generate_neg_sample(self, h, r):
        n = 0
        neg_samples = []
        while n < self.neg_ratio:
            negative_sample = np.random.choice(self.len_tail, size=self.neg_ratio * 2, replace=False)
            mask = np.in1d(
                self.tail[negative_sample],
                self.all_triple[(h, r)],
                assume_unique=True,
                invert=True
            )
            negative_sample = self.tail[negative_sample[mask]]
            neg_samples.append(negative_sample)
            n += negative_sample.size
        neg_samples = np.concatenate(neg_samples)[:self.neg_ratio]
        return neg_samples

    def __getitem__(self, index) -> (torch.tensor, torch.tensor, torch.tensor, torch.tensor):
        """

        :param index:
        :return: [head * (1 + neg_ratio)], [pos_tail, neg_tail * neg_ratio], label[1, 0 * neg_ratio]
        """
        positive_sample = self.sub_triple[index]
        negative_samples = self.generate_neg_sample(
            positive_sample[0], positive_sample[1])
        head = [positive_sample[0]]
        tail = [positive_sample[2]]
        positive_label = [1]
        negative_label = [0 for _ in range(self.neg_ratio)]
        return (torch.tensor(head), torch.tensor(tail), torch.tensor(negative_samples),
                torch.tensor(positive_label, dtype=torch.float), torch.tensor(negative_label, dtype=torch.float))

    def __len__(self):
        return len(self.sub_triple)


class Sampler(object):
    def __init__(self, triples, entity_num, edge_type_num):
        super(Sampler, self).__init__()
        self.triples = triples
        self.head_r_tail_dict = self.__generate_head_r_tail_dict()
        self.max_neighbor_num = self.__calc_max_neighbor_num()
        self.entity_num = entity_num
        self.edge_type_num = edge_type_num

    def __generate_head_r_tail_dict(self):
        # generate a head_tail dict which key is head_id,value is list of tail_id.
        head_tail = defaultdict(list)
        triples = self.triples
        for triple in triples:
            h, r, t = triple
            head_tail[h].append((r, t))
            head_tail[t].append((r, h))

        return head_tail

    def __calc_max_neighbor_num(self):
        num = max(list(map(len, self.head_r_tail_dict.values())))
        return num

    def sampling_all(self, node_ids):
        return self.sampling(node_ids, self.max_neighbor_num)

    def multi_hop_sampling_all(self, node_ids, num_hop):

        pass

    def sampling(self, node_ids, num):
        node_result = []
        edge_result = []
        for node_id in node_ids:
            try:
                tails = copy.deepcopy(self.head_r_tail_dict[node_id])
            except KeyError:
                self.head_r_tail_dict[node_id] = [(self.edge_type_num, self.entity_num)]
                tails = copy.deepcopy(self.head_r_tail_dict[node_id])
            if num <= len(tails):
                tails = random.sample(tails, k=num)
            else:
                tails.extend([(self.edge_type_num, self.entity_num)] * (num - len(tails)))

            res = np.array(tails)
            node_result.append(res[:, 1])
            edge_result.append(res[:, 0])

        return np.asarray(node_result).flatten(), np.asarray(edge_result).flatten()

    def multi_hop_sampling(self, node_ids, sampling_list):
        """

        :param node_ids:list of node_id
        :param sampling_list: a list of num.Each number means the number of neighbor we want to sample.
        :return: a list contains node_id and its k-hop neighbor node_id
        """
        node_result = [np.array(node_ids)]
        edge_result = []
        for k, num in enumerate(sampling_list):
            hop_k_node_result, hop_k_edge_result = self.sampling(node_result[k], num)
            node_result.append(hop_k_node_result)
            edge_result.append(hop_k_edge_result)
        return node_result, edge_result


class OneShotIterator(object):
    def __init__(self, kg_data_loader):
        self.iterator_data = self.one_shot_iterator(kg_data_loader)

    def __next__(self):

        data = next(self.iterator_data)

        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        """
        Transform a PyTorch Dataloader into python iterator
        """
        while True:
            for data in dataloader:
                yield data
