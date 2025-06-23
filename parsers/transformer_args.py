import argparse

import torch

import transformer_config


def get_transformer_args(arg_list=None):
    parser = argparse.ArgumentParser()

    # used for auto_train.py to split training data
    parser.add_argument('-data', type=str, help='dataset json file')
    parser.add_argument('-train_ratio', type=float, default=0.6,
                        help='proportion of data used for training (default 0.6)')
    parser.add_argument('-dev_ratio', type=float, default=0.2,
                        help='proportion of data used for validation (default 0.2)')
    parser.add_argument('-test_ratio', type=float, default=0,
                        help='proportion of data used for testing (default 1 - train_ratio - dev_ratio)')
    parser.add_argument('-use_gp', action='store_true', default=False)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-data_seed', type=str, default=None,
                        help='not none for fix the dataset with the data generation random seed')
    parser.add_argument('-target', type=str, choices=['eff', 'vout', 'eff_vout', 'reward', 'valid'], default='eff')
    parser.add_argument('-save_model', type=str, default=None, help='save the trained model to this file, if provided')
    parser.add_argument('-target_vout', type=float, default=50, help='target output voltage when using reward')

    # mostly fixed
    parser.add_argument('-ground_truth', type=str, choices=['simulation', 'analytic'], default='simulation')
    parser.add_argument('-test_ground_truth', type=str, choices=['simulation', 'analytic'], default='simulation')

    parser.add_argument('-split_by', type=str, choices=['data', 'topo'], default='data')

    parser.add_argument('-circuit_threshold', type=float, default=0.6, help='good circuit reward threshold')
    parser.add_argument('-extra_datasets', type=str, nargs='+', default=['dataset_4_anal_label.json',
                                                                         'dataset_3_valid.json'],
                        help='extra datasets')

    parser.add_argument('-select_cond', type=str, default='none', choices=['fix_duty', 'none'])
    parser.add_argument('-no_cuda', action='store_true', default=False)
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('-add_attribute', action='store_true')  # dummy

    # different encoding methods for Transformer encoders, use LSTM by default
    #
    # absolute: use conventional position encoding
    # absolute_2d: use (row embedding, column embedding)
    # relative: use relative embedding
    #   idea: https://arxiv.org/pdf/1803.02155.pdf
    #   implementation: https://arxiv.org/pdf/1809.04281.pdf
    # hierarchical_transformer: use transformer to generate a path embedding
    # lstm: use lstm to generate a path embedding (following the diagram in DOE report)
    # none: for debugging, no encoding is used (so position information is completely lost)
    parser.add_argument('-encoding', type=str, default='lstm',
                        choices=['absolute', 'absolute_2d', 'relative', 'learnable', 'hierarchical_transformer', 'lstm',
                                 'none'])
    # ways to encode the duty cycle
    # mlp: use it as input in MLP
    # path: compute an embedding vector for it, and concatenate with path embedding
    # none: ignore duty cycle information
    parser.add_argument('-duty_encoding', type=str, default='path',
                        choices=['mlp', 'path', 'none'])
    # by default, don't use gp

    parser.add_argument('-data_train', type=str, default="")
    parser.add_argument('-data_dev', type=str, default="")
    parser.add_argument('-data_test', type=str, default="")
    # if this vocab file unless using old path-based vocab
    parser.add_argument('-vocab', default='compwise_vocab.json', type=str)

    parser.add_argument('-epoch', type=int, default=5)
    parser.add_argument('-batch_size', type=int, default=512)

    parser.add_argument('-d_model', type=int, default=128)

    parser.add_argument('-n_heads', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=2)
    parser.add_argument('-n_warmup_steps', type=int, default=20)
    parser.add_argument('-mlp_layers', type=int, nargs='+', default=[64, 16])

    parser.add_argument('-dropout', type=float, default=0.1)

    parser.add_argument('-no_log', action='store_true', default=True)
    parser.add_argument('-plot_outliers', action='store_true',
                        help='plot the cases where ground truth and surrogate model disagree')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-num_workers', type=int, default=1)

    # maximum number of paths in a topology
    parser.add_argument('-max_seq_len', type=int, default=transformer_config.max_path_num)
    # maximum number of devices in a path
    parser.add_argument('-attribute_len', type=int, default=transformer_config.max_path_len)

    parser.add_argument('-pretrained_model', type=str, default=None)
    parser.add_argument('-pretrained_validity_model', type=str, default=None)

    parser.add_argument('-patience', type=int, default=2)
    parser.add_argument('-beam_size', type=int, default=5)  # dummy

    if arg_list is not None:
        args = parser.parse_args(arg_list)
    else:
        args = parser.parse_args()

    if args.data is not None:
        # if provided raw data file, set train, dev, test files automatically
        file_name = args.data.split('.')[0]

        args.data_train = file_name + '_train_' + str(args.train_ratio) + '_' + str(args.seed) + '.json'
        args.data_dev = file_name + '_dev_' + str(args.train_ratio) + '_' + str(args.seed) + '.json'
        args.data_test = file_name + '_test_' + str(args.train_ratio) + '_' + str(args.seed) + '.json'

    if args.target == 'valid' and args.duty_encoding != 'none':
        print("Assumed validity of topologies does not depend on duty cycles. Setting duty_encoding to none.")
        args.duty_encoding = 'none'

    args.use_log = not args.no_log

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    args.d_word_vec = args.d_model

    args.load_weights = (args.pretrained_model is not None)

    args.device = torch.device('cuda' if args.cuda else 'cpu')

    return args
