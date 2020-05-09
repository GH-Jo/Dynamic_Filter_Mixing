"""
    train
    ~~~~~

    
"""


from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import random
import time
import os
import argparse
import logging
import glob
import sys

from vgg import *
from utils import *
import module

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1'):
        return True
    elif v.lower() in ('false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--lr', default=0.1, type=float, 
                            help='learning rate')
parser.add_argument('--gamma', default=0.1, type=float)
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--test_batch_size', type = int, default = 1024)
parser.add_argument('--arch', default='dfm_vgg16', type = str, 
                            help='architecture')
parser.add_argument('--gpu', default=[0], type = list)
parser.add_argument('--dataset', default = 'c10', type = str)
parser.add_argument('--log_interval', default = 100, type = int)
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                            help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--seed', type = int, default = 2)
parser.add_argument('--epochs', type = int, default = 400)
parser.add_argument('--warmup', type = int, default = 10)
parser.add_argument('--balance_weight', type = float, default = 1e-4)

# ---  Newly added in DFM ---
parser.add_argument('--experiment', type = int)

parser.add_argument('--channel_reduction', type=int, default=2)
parser.add_argument('--base_reduction', type=int, default=4)
parser.add_argument('--hidden_reduction', type=int, default=8)
parser.add_argument('--att_tau', type = int, default = 30)
parser.add_argument('--att_tau_epochs', type = int, default = 10)

parser.add_argument('--lr_decay_type', type = str, default='cosine',
                        help='[ cosine | cosine_warm | multistep ]')
parser.add_argument('--restart_epochs', type = int, default=400)

parser.add_argument('--kd', type=str2bool, default=False)
parser.add_argument('--kd_lambda', type=float, default=0.5)
parser.add_argument('--kd_tau', type=float, default=3)

parser.add_argument('--p_att', type=float, default=0,
                        help='dropout rate for attention layers')
parser.add_argument('--p_cls', type=float, default=0,
                        help='dropout rate for classifier layers')

parser.add_argument('--dfm_start', type=str, default='entire',
                        help='[ entire | 2 .. 15 ]')

args = parser.parse_args()
args.save = 'logs/{0}/fm-#{1}-{2}{3}-{4}-{5}-{6}'.format(
                args.dataset,
                args.experiment,
                '-kd-' if args.kd else '',
                args.channel_reduction,
                args.base_reduction,
                args.hidden_reduction,
                time.strftime("%Y%m%d-%H%M%S"))
print(args)

create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(g) for g in args.gpu])

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2023, 0.1994, 0.2010)),
])

if args.dataset == 'c10':
    trainset = torchvision.datasets.CIFAR10(root='/home/lkj004124/data', train=True, \
                              download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='/home/lkj004124/data', train=False, \
                              download=True, transform=transform_test)
    n_classes = 10

trainloader = torch.utils.data.DataLoader(trainset, \
                          batch_size=args.batch_size, \
                          shuffle=True, pin_memory = True, num_workers=0)

testloader = torch.utils.data.DataLoader(testset, \
                          batch_size=args.test_batch_size, \
                          shuffle=False, pin_memory = True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', \
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
if args.arch == 'dfm_vgg16':
    model = dfm_vgg16(args.arch, n_classes,
                      args.channel_reduction,
                      args.base_reduction, 
                      args.hidden_reduction,
                      args.att_tau, 
                      args.p_att,
                      args.p_cls,
                      args.dfm_start)
    print(model)
if args.kd:
    model_kd = vgg16(args.arch, n_classes)
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model_kd.load_state_dict(checkpoint.state_dict())

logging.info("args = %s", args)
model = model.cuda()
if args.kd:
    model_kd = model_kd.cuda()

# loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD([p for n, p in model.named_parameters() \
                             if p.requires_grad ], \
                      lr=args.lr, momentum = args.momentum, \
                      weight_decay = args.weight_decay)
if (args.lr_decay_type.lower() == 'cosine') \
        or (args.lr_decay_type.lower() =='cos'):
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
            args.epochs)

elif ('warm' in args.lr_decay.lower()) or ('restart' in args.lr_decay.lower()):
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
            T_0=args.restart_epochs)

elif ('multistep' in args.lr_decay.lower()) :
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
            [50, 100, 150, 200, 300], gamma=0.1)
else:
    raise argparse.ArgumentTypeError('Wronge lr_decay_type entered: {}'.\
             format(args.lr_decay_type))
    

def train(epoch, args):
    logging.info('\nEpoch: %d, Learning rate: %.8f', \
                    epoch, scheduler.get_lr()[0])
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # assign input and target
        inputs, targets = inputs.cuda(), targets.cuda()
        data_time = time.time()

        # forward and backward
        outputs = model(inputs)

        hard_criterion = criterion(outputs, targets)
        if args.kd:
            soft_target = model_kd(inputs)
            loss = args.kd_lambda * soft_criterion(outputs, soft_target, args.kd_tau) \
                    + (1-args.kd_lambda) * hard_criterion
        else:
            loss = hard_criterion
        loss.backward()

        # optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # loss and number of correct result 
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_loss += loss.item()
        model_time = time.time()

        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: %d\t Process: %d/%d\t ' + \
                        'Loss: %.06f\t Data Time: %.03f s\t' +  \
                        'Model Time: %.03f s',   # \t Memory %.03fMB', 
                epoch, batch_idx * len(inputs), 
                len(trainloader.dataset), 
                train_loss/batch_size/batch_idx, data_time - end, 
                model_time - data_time)

        end = time.time()

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct, test_loss / len(testloader)

if __name__ == '__main__':
    cudnn.benchmark = True
    torch.cuda.manual_seed(args.seed)
    cudnn.enabled = True
    torch.manual_seed(args.seed)
    
    max_correct = 0
    for epoch in range(args.epochs):
        if epoch == args.warmup:
            optimizer = optim.SGD(\
                    [p for n, p in model.named_parameters() \
                           if p.requires_grad],
                            #and 'combination' not in n], 
                    lr=args.lr, momentum = args.momentum, 
                    weight_decay = args.weight_decay)

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                              args.epochs-args.warmup)
        train(epoch, args)
        scheduler.step()
        correct, loss = test(epoch)
        if correct > max_correct:
            max_correct = correct
            torch.save(model, os.path.join(args.save, 'weights.pth'))
        logging.info('Epoch %d Correct: %d,'\
                      +'Max Correct %d, Loss %.06f', \
                      epoch, correct, max_correct, loss)
        if epoch < args.att_tau_epochs:
            for layer in model.features.children():
                if isinstance(layer, module.DFMConv2d):
                    layer.tau_decay(args.att_tau, args.att_tau_epochs)


