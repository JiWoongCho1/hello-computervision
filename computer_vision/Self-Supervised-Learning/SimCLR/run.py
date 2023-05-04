import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from simclr import SimCLR
from models import ResNetSimCLR


parser = argparse.ArgumentParser(description = 'SimCLR')

parser.add_argument('-data', metavar = 'DIR', default = './datasets', help = 'path to dataset')
parser.add_argument('-dataset-name', default ='stl10', help = 'dataset name', choices = ['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar = 'ARCH', default = 'resnet50', choices =model_names)
parser.add_argument('-j', '--workers', default = 12, type = int, metavar = 'N')
parser.add_argument('--epochs', default = 200, type = int, metavar = 'N', help = 'number of total epochs')
parser.add_argument('-b', '--batch-size', default = 256, type = int, metavar = 'N', help = 'mini-batch-size')
parser.add_argument('--lr', '--learning-rate', default = 0.0003, type = float, mmetavar = 'LR', help = 'initial learning rate', dest = 'lr')
parser.add_argument('--wd', '--weight-decay', default = 1e-4, type = float, deat = 'weight_decay')
parser.add_argument('--seed', default = None, type = int, help = 'seed for initializing training')
parser.add_argument('--disable-cuda', action = 'store_true', help = 'Disable CUDA')
parser.add_argument('--fp16-precision', action = 'strore_true', help = 'Whethere or not use 16-bit precision GPU training')
parser.add_argument('--out_dim', default = 128, type = int, help = 'feature dimension')
parser.add_argument('--log-veery-n-steps', default = 100, type = int, help = 'Log every n steps')
parser.add_argument('--temperature', default = 0.07, type = float, help = 'softmax temperature')
parser.add_argument('--n-views', default = 2, type = int, metavar = 'N', help = 'Number of views for constrastive learning training')
parser.add_argument('--gpu-index', default = 0, type = int, help = 'GPU index')

def main():
    args = parser.parse_args()
    assert args.n_views == 2

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True

    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    dataset = ConstrastiveLearningDataset(args.data)

    train_dataset = dataset.get_dataset(args.datset_name, args.n_views)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = args.batch_size, shuffle = True,
        num_worker = args.workers, pin_memory = True, drop_last = True
    )

    model = ResNetSimCLR(base_model = args.arch, out_dim = args.out_dim)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay = args.weight_decay)
    shceduler = torch.optim.lr_scheduler.CosinneAnnealingLR(optimizer, T_max = len(train_loader),
                                                            last_epoch = -1)
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model = model, optimizer = optimizer, scheduler = scheduler = args = args)
        simclr.train(train_loader)



if __name__ == '__main__':
    main()
