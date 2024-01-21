import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="456")
    parser.add_argument('--gpu', action='store_true', help='enable gpu')
    parser.add_argument('--save_model', action='store_true', help='save_model')

    parser.add_argument('--enable_gumbel', action='store_true', help='enable gumbel-softmax')
    parser.add_argument('--enable_augmentation', action='store_true', help='enable_augmentation')
    parser.add_argument('--save_path', nargs='?', default='./log/result.pkl', help='Input save path.')
    parser.add_argument('--data_path', nargs='?', default='./data/Hetionet',
                        help='Input data path.')
    parser.add_argument('--score_fun', nargs='?', default='dot', help='Input data path.')
    parser.add_argument('--embed_dim', type=int, default=64,
                        help="the embedding size entity and relation")
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--valid_step', type=int, default=1)
    parser.add_argument('--hop', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--layer_num', type=int, default=1,
                        help="the layer num")
    parser.add_argument('--neg_ratio', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.002,
                        help="the learning rate")
    parser.add_argument('--tau', type=float, default=1.3,
                        help="the learning rate")
    parser.add_argument('--amplitude', type=float, default=0.6,
                        help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-4,
                        help="the weight decay for l2 regulation")
    parser.add_argument('--dropout', type=float, default=0.2,
                        help="using the dropout ratio")
    parser.add_argument('--head_num', type=int, default=2,
                        help="the head num")
    return parser.parse_args()
