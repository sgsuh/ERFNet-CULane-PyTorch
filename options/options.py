import argparse
parser = argparse.ArgumentParser(description = "PyTorch implementation of Semantic Segmentation")

parser.add_argument('--dataset', 
                    type = str, 
                    default = 'CULane', 
                    choices = ['VOCAug', 'VOC2012', 'COCO', 'Cityscapes', 'ApolloScape', 'CULane'])
parser.add_argument('--method', 
                    type = str, 
                    default = 'ERFNet', 
                    choices = ['FCN', 'DeepLab', 'DeepLab3', 'PSPNet', 'ERFNet'])
parser.add_argument("--data_path",
                    type=str,
                    default="")
parser.add_argument('--train_list', 
                    type = str, 
                    default = 'train_gt')
parser.add_argument('--val_list', 
                    type = str, 
                    default = 'val_gt')

# ========================= Model Configs ==========================
parser.add_argument('--arch', 
                    type = str, 
                    default = "resnet101")
parser.add_argument('--dropout', 
                    '--do', 
                    default = 0.1, 
                    type = float, 
                    metavar = 'DO', 
                    help = 'dropout ratio (default: 0.1)')
parser.add_argument('--train_size', 
                    default = 840, 
                    type = int, 
                    metavar = 'L', 
                    help = 'size of training patches (default: 473)')
parser.add_argument('--test_size', 
                    default = 840, 
                    type = int, 
                    metavar = 'L', 
                    help = 'size of testing patches (default: 513)')
parser.add_argument('--img_height', 
                    default = 270, 
                    type = int, 
                    metavar = 'L', 
                    help = 'height of input images (default: 208)')
parser.add_argument('--img_width', 
                    default = 960, 
                    type = int, 
                    metavar = 'L', 
                    help = 'width of input images (default: 976)')
parser.add_argument('--local_rank', 
                    type = int, 
                    default = 0) # distributed data parallel

# =========== for test ============
parser.add_argument('--loss', 
                    type = str, 
                    default = 'ce', 
                    choices = ['ce', 'focal'])
parser.add_argument('--optm', 
                    type = str, 
                    default = 'sgd', 
                    choices = ['sgd', 'adam', 'adamw'])
parser.add_argument('--schd', 
                    type = str, 
                    default = 'linear', 
                    choices = ['linear', 'cosine', 'plateau'])
parser.add_argument('--mem', 
                    type = str, 
                    default = 'False')
parser.add_argument('--seg_bg_w', 
                    default = 0.4, 
                    type = float)
parser.add_argument('--est_bg_w', 
                    default = 0.4, 
                    type = float)
parser.add_argument('--shortcut', 
                    type = str, 
                    default = 'False')
parser.add_argument('--server', 
                    type = str, 
                    default = 'True')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', 
                    default = 200, 
                    type = int, 
                    metavar = 'N', 
                    help = 'number of total epochs to run')
parser.add_argument('-b', 
                    '--batch-size', 
                    default = 64, 
                    type = int, 
                    metavar = 'N', 
                    help = 'mini-batch size (default: 256)')
parser.add_argument('--lr', 
                    '--learning-rate', 
                    default = 0.02, 
                    type = float, 
                    metavar = 'LR', 
                    help = 'initial learning rate')
parser.add_argument('--lr_steps', 
                    default = [10, 20], 
                    type = float, 
                    nargs = "+", 
                    metavar = 'LRSteps', 
                    help = 'epochs to decay learning rate by 10')
parser.add_argument('--momentum', 
                    default = 0.9, 
                    type = float, 
                    metavar = 'M', 
                    help = 'momentum')
parser.add_argument('--weight-decay', 
                    '--wd', 
                    default = 1e-4, 
                    type = float, 
                    metavar = 'W', 
                    help = 'weight decay (default: 5e-4)')

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', 
                    '-p', 
                    default = 1, 
                    type = int, 
                    metavar = 'N', 
                    help = 'print frequency (default: 10)')
parser.add_argument('--eval-freq', 
                    '-ef', 
                    default = 1, 
                    type = int, 
                    metavar = 'N', 
                    help = 'evaluation frequency (default: 5)')

# ========================= Runtime Configs ==========================
parser.add_argument('-j', 
                    '--workers', 
                    default = 1, 
                    type = int, 
                    metavar = 'N', 
                    help = 'number of data loading workers (default: 16)')
parser.add_argument('--resume', 
                    type = str, 
                    default = 'erfnet_v2f_0.pth', 
                    metavar = 'PATH', 
                    help = 'path to latest checkpoint (default: none)')
parser.add_argument('--weight', 
                    default = '', 
                    type = str, 
                    metavar = 'PATH', 
                    help = 'path to initial weight (default: none)')
parser.add_argument('-e', 
                    '--evaluate', 
                    dest = 'evaluate', 
                    action = 'store_true', 
                    help = 'evaluate model on validation set') # true
parser.add_argument('--snapshot_pref', 
                    type = str, 
                    default = "")
parser.add_argument('--start-epoch', 
                    default = 0, 
                    type = int, 
                    metavar = 'N', 
                    help = 'manual epoch number (useful on restarts)')
parser.add_argument('--gpus', 
                    nargs = '+', 
                    type = int, 
                    default = [0, 1, 2, 3])


parser.add_argument('--seg_mode', 
                    type = str, 
                    default = 'class', 
                    choices = ['class', 'branch'])

parser.add_argument('--encoder', 
                    type = str, 
                    default = '')
parser.add_argument('--est_decoder', 
                    type = str, 
                    default = '')
parser.add_argument('--seg_decoder', 
                    type = str, 
                    default = '')