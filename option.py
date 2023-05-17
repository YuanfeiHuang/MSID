import argparse, torch

parser = argparse.ArgumentParser(
    description='Multi-scale Information Distillation Network for Efficient Image Super-Resolution')

# Hardware specifications
parser.add_argument('--cuda', default=True, action='store_true', help='Use cuda?')
parser.add_argument('--GPU_ID', type=str, default=0, help='GPUs id')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loading')
parser.add_argument('--seed', type=int, default=0, help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='data/Datasets/', help='dataset directory')
parser.add_argument('--data_train', type=str, default=['DF2K'], help='train dataset name')
parser.add_argument('--data_test', type=str, default=['Set5'], help='test datasets name')
parser.add_argument('--n_train', type=int, default=[3450], help='number of training samples')
parser.add_argument('--scale', type=int, default=3, help='scale factor')
parser.add_argument('--n_colors', type=int, default=3, help='RGB color images')
parser.add_argument('--value_range', type=int, default=255, help='maximum value of RGB')
parser.add_argument("--store_in_ram", default=True, action="store_true", help='store the training set in RAM')

# Training/Testing specifications
parser.add_argument('--train', type=str, default='test', help='train | test')
parser.add_argument('--iter_epoch', type=int, default=20, help='iteration in each epoch')
parser.add_argument('--start_epoch', default=-1, type=int, help='start epoch for training')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--patch_size', type=int, default=48, help='spatial resolution of training samples')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
parser.add_argument('--model_path', type=str, default='', help='model path')
parser.add_argument('--resume', type=str, default='', help='checkpoint path')

# Optimization specifications
parser.add_argument('--optimizer', default=torch.optim.Adam, help='optimizer')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--lr_gamma_1', type=int, default=80, help='learning rate decay per N epochs')
parser.add_argument('--lr_gamma_2', type=float, default=0.5, help='gamma for decay')
args = parser.parse_args()