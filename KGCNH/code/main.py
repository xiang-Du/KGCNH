import os
from parse import parse_args
from data_loader import DataProcessor, DrugDiseaseDataSet
from model import Model
import torch
from torch.utils.data import DataLoader
from procedure import train, test
from utils import print_config, set_seed, generate_bi_coo_matrix, save_model, exclude_isolation_point, \
    generate_directed_coo_matrix
# from utils import comp
from sklearn.model_selection import KFold
from torch.optim import Adam, lr_scheduler
import numpy as np
import dgl


def main():
    # seed 12, 34, 42, 43, 61, 70, 83, 1024, 2014, 2047
    args = parse_args()
    seed = args.seed
    set_seed(1897)

    if torch.cuda.is_available() and args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    if args.hop == 1:
        hop = args.hop
    else:
        hop = None
    score_fun = args.score_fun
    epoch = args.epoch
    data_path = args.data_path
    data = DataProcessor(data_path, hop)
    entity_num = data.get_node_num()
    relation_num = data.get_relation_num()
    drug_disease_num = data.drug_disease_num
    drug_num = data.drug_num
    disease_num = data.disease_num
    data_info = dict()
    data_info['device'] = device
    data_info['entity_num'] = entity_num
    data_info['drug_disease_num'] = drug_disease_num
    data_info['disease_num'] = disease_num
    data_info['drug_num'] = drug_num
    data_info['relation_num'] = relation_num
    data_info['score_fun'] = score_fun
    data_info['seed'] = seed
    # drug treats disease triple
    dd_triples = data.load_data('treats')
    # knowledge triple
    kg_triples = data.load_data('others')
    model_args = (args.layer_num, args.head_num)
    data_info['model_args'] = model_args
    data_info['dd_triple_num'] = len(dd_triples)
    data_info['kg_triple_num'] = len(kg_triples)
    print_config(data_info, args)
    save_path = args.save_path
    save_model_sign = args.save_model
    model_save_dir = os.path.split(save_path)[0]
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_dir = os.path.join(model_save_dir, str(seed))
    if not os.path.exists(model_save_dir) and save_model_sign:
        os.makedirs(model_save_dir)
    avg_auc, avg_aupr = 0, 0
    if 'DrugBank' in data_path:
        add_train = True
    else:
        add_train = False
    # 10-fold validation
    k_fold = KFold(n_splits=10, shuffle=True, random_state=seed)
    for idx, (train_idx, valid_idx) in enumerate(k_fold.split(dd_triples)):
        train_set = [tuple(i) for i in np.array(dd_triples)[train_idx].tolist()]
        valid_set = [tuple(i) for i in np.array(dd_triples)[valid_idx].tolist()]
        # exclude the isolation triple in valid_set and add it to train_set
        valid_set, exclusion_list = exclude_isolation_point(train_set, valid_set)
        train_set = train_set + exclusion_list
        train_set_len = len(train_set)
        valid_set_len = len(valid_set)
        print('remove {} triple from valid_set, train_set {}, valid_set {}'.format(len(exclusion_list), train_set_len,
                                                                                   valid_set_len))
        # fix valid neg sample
        dd_valid_set = DrugDiseaseDataSet(valid_set, dd_triples, 1)
        valid_data_loader = DataLoader(dd_valid_set, batch_size=valid_set_len, shuffle=True)
        h, t, neg_t, pos_label, neg_label = next(iter(valid_data_loader))
        valid_data_loader = ((h, t, neg_t, pos_label, neg_label),)
        h = h.expand(neg_t.shape).view(-1).to('cpu').tolist()
        neg_t = neg_t.view(-1).to('cpu').tolist()
        r = dd_triples[0][1]
        valid_neg_sample = [(h[i], r, neg_t[i]) for i in range(len(h))]
        if add_train:
            train_kg = kg_triples + train_set
        else:
            train_kg = kg_triples

        train_graph = train_kg

        # generate kg_graph
        if hop != 1:
            kg_coo, kg_relation = generate_bi_coo_matrix(train_kg)
            augment_coo, augment_relation = generate_bi_coo_matrix(train_graph)
        else:
            kg_coo, kg_relation = generate_directed_coo_matrix(train_kg, drug_disease_num)
            augment_coo, augment_relation = generate_directed_coo_matrix(train_graph, drug_disease_num)
        kg_graph = dgl.graph(kg_coo).to(device)
        augment_graph = dgl.graph(augment_coo).to(device)
        # add valid neg sample into train_set to avoid generating same neg sample
        dd_train_set = DrugDiseaseDataSet(train_set, dd_triples + valid_neg_sample, args.neg_ratio)
        train_data_loader = DataLoader(dd_train_set, batch_size=train_set_len, shuffle=True)
        print(len(kg_relation))
        #############################################################
        # flag = None
        # for i in range(100):
        #     for data in train_data_loader:
        #         h, t, neg_t, pos_label, neg_label = data
        #         print(neg_t[-1, -10:])
        #         h = h.expand(neg_t.shape).contiguous().view(-1).to('cpu').tolist()
        #         neg_t = neg_t.view(-1).to('cpu').tolist()
        #         r = dd_triples[0][1]
        #         train_neg_sample = [(h[i], r, neg_t[i]) for i in range(len(h))]
        #         flag = comp(valid_neg_sample + valid_set, train_neg_sample + train_set)
        #         if flag:
        #             break
        # if flag:
        #     break
        # print(flag)
        # break
        ############################################################
        model = Model(entity_num, drug_num, disease_num, relation_num, args.embed_dim,
                      model_args, args.enable_augmentation,
                      (args.enable_gumbel, args.tau, args.amplitude / args.epoch), args.decay, args.dropout,
                      score_fun).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[0.3 * epoch, 0.6 * epoch, 0.8 * epoch], gamma=0.8)
        for name, para in model.named_parameters():
            print(name)
        print(model)

        file_name = 'result' + str(idx) + '.pkl'
        model_save_path = os.path.join(model_save_dir, file_name)

        result = {'best_auc': 0, 'best_aupr': 0, 'valid_data_loader': valid_data_loader}
        for i in range(epoch):

            loss, reg_loss = train(model, kg_graph, augment_graph, kg_relation, augment_relation,
                                   train_data_loader, optimizer, device)

            if (i + 1) % args.valid_step == 0:
                print("epoch:{},loss:{}, reg_loss:{}".format(i + 1, loss, reg_loss))
                auc, aupr, all_score, all_label, embed = test(model, kg_graph, augment_graph, kg_relation,
                                                              augment_relation,
                                                              valid_data_loader, device)

                # train_auc, train_aupr, _, _ = test(model, kg_graph, augment_graph, kg_relation, augment_relation,
                #                                    train_data_loader, device)
                #
                # print("epoch:{},train_auc:{}, train_aupr:{},auc:{}, aupr:{}".format(i + 1, train_auc, train_aupr, auc,
                #                                                                     aupr))
                print("epoch:{},auc:{}, aupr:{}".format(i + 1, auc, aupr))
                if result['best_auc'] < auc:
                    result['epoch'] = i + 1
                    result['best_auc'] = auc
                    result['best_aupr'] = aupr
                    result['best_score'] = all_score
                    result['label'] = all_label
                    if save_model_sign:
                        res = {'model': model, 'embed': embed, 'valid_data_loader': valid_data_loader}
                        save_model(res, model_save_path, 'wb')

            scheduler.step()
        save_model(result, save_path)
        avg_auc += result['best_auc']
        avg_aupr += result['best_aupr']
        print(idx, result['epoch'], result['best_auc'], result['best_aupr'])
    print('avg_auc:', avg_auc / 10, "avg_aupr:", avg_aupr / 10)


if __name__ == '__main__':
    main()
