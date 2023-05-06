import argparse
import torch
from utils.utils import *

parser = argparse.ArgumentParser(description='BINARY')

parser.add_argument('--data_set', default='DIV2K',type=str,
                    help='Enables debug mode')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=8,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
torch.cuda.set_device(2)

# Data specifications
parser.add_argument('--scale', type=int, default=2,
                    help='super resolution scale')
parser.add_argument('--data_type', type=str, default='npy',
                    help='data type : test:img , train:npy')
parser.add_argument('--dataroot_HR', type=str, default='/home/gpu/data/HR/HR_x2',
                    help='train high-resolution dataset')
parser.add_argument('--dataroot_LR', type=str, default='/home/gpu/data/LR/LR_x2',
                    help='train low-resolution dataset')
parser.add_argument('--dataroot_HR_val', type=str, default="/home/gpu/Data/benchmark/Set5/HR/",
                    help='train high-resolution dataset')
parser.add_argument('--dataroot_LR_val', type=str, default="/home/gpu/Data/benchmark/Set5/LR_bicubic/X2/",
                    help='train low-resolution dataset')
parser.add_argument('--data_test', type=str, default='B100',
                    help='test dataset name')
parser.add_argument('--patch_size', type=int, default=48,
                    help='input patch size')
parser.add_argument('--noise', type=str, default='.',
                    help='train low-resolution dataset')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')

# Model specifications
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of feature maps')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--n_resblocks', type=int, default=32,
                    help='number of residual blocks')
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')


# Training specifications
parser.add_argument('--pre_model', default='/data/JXR/2021-MM-TIP/3/experiment/2022_6_30_wavlet/CS_scenes_best.pth',type=str, metavar='pre_model path',
                    help='pre_model path')
parser.add_argument("--save_dir", type=str, default='../experiment/2022_0818',
                    help="Where to save the model.")
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train')
parser.add_argument('--print_every', type=int, default=200,
                    help='how many batches to wait before logging training status')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--decay', type=int, default=150,
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--weight_decay', type=float, default=1e-6,
                    help='weight decay')

args = parser.parse_args()

args.log_path = args.save_dir + '/log'

log_init(args.log_path, args.data_set)
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(args.log_path):
    os.makedirs(args.log_path)
logger_path = args.log_path + '/tensorboard/'

for key, val in args._get_kwargs():
    logging.info(key+' : '+str(val))

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

