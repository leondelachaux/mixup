#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models
from utils import progress_bar

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="ResNet18", type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    torch.manual_seed(args.seed)

# Data
# print('==> Preparing data..')
# if args.augment:
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465),
#                              (0.2023, 0.1994, 0.2010)),
#     ])
# else:
#     transform_train = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465),
#                              (0.2023, 0.1994, 0.2010)),
#     ])


# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# trainset = datasets.CIFAR10(root='~/data', train=True, download=True,
#                             transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset,
#                                           batch_size=args.batch_size,
#                                           shuffle=True, num_workers=8)

# testset = datasets.CIFAR10(root='~/data', train=False, download=True,
#                            transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100,
#                                          shuffle=False, num_workers=8)


  torch.manual_seed(1)
  np.random.seed(1)

  # Load datasets
  train_transform = transforms.Compose(
      [transforms.RandomHorizontalFlip(),
       transforms.RandomCrop(32, padding=4)])
  preprocess = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize([0.5] * 3, [0.5] * 3)])
  test_transform = preprocess

  if args.dataset == 'cifar10':
    train_data = datasets.CIFAR10(
        './data/cifar', train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR10(
        './data/cifar', train=False, transform=test_transform, download=True)
    base_c_path = './data/cifar/CIFAR-10-C/'
    num_classes = 10
  else:
    train_data = datasets.CIFAR100(
        './data/cifar', train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR100(
        './data/cifar', train=False, transform=test_transform, download=True)
    base_c_path = './data/cifar/CIFAR-100-C/'
    num_classes = 100

  # train_data = AugMixDataset(train_data, preprocess, args.no_jsd)
  # train_loader = torch.utils.data.DataLoader(
  #     train_data,
  #     batch_size=args.batch_size,
  #     shuffle=True,
  #     num_workers=args.num_workers,
  #     pin_memory=True)

  # test_loader = torch.utils.data.DataLoader(
  #     test_data,
  #     batch_size=args.eval_batch_size,
  #     shuffle=False,
  #     num_workers=args.num_workers,
  #     pin_memory=True)

  # # Create model
  # if args.model == 'densenet':
  #   net = densenet(num_classes=num_classes)
  # elif args.model == 'wrn':
  #   net = WideResNet(args.layers, num_classes, args.widen_factor, args.droprate)
  # elif args.model == 'allconv':
  #   net = AllConvNet(num_classes)
  # elif args.model == 'resnext':
  #   net = resnext29(num_classes=num_classes)

  # optimizer = torch.optim.SGD(
  #     net.parameters(),
  #     args.learning_rate,
  #     momentum=args.momentum,
  #     weight_decay=args.decay,
  #     nesterov=True)

  # # Distribute model across all visible GPUs
  # net = torch.nn.DataParallel(net).cuda()
  # cudnn.benchmark = True

  # start_epoch = 0

  # if args.resume:
  #   if os.path.isfile(args.resume):
  #     checkpoint = torch.load(args.resume)
  #     start_epoch = checkpoint['epoch'] + 1
  #     best_acc = checkpoint['best_acc']
  #     net.load_state_dict(checkpoint['state_dict'])
  #     optimizer.load_state_dict(checkpoint['optimizer'])
  #     print('Model restored from epoch:', start_epoch)

  # if args.evaluate:
  #   # Evaluate clean accuracy first because test_c mutates underlying data
  #   test_loss, test_acc = test(net, test_loader)
  #   print('Clean\n\tTest Loss {:.3f} | Test Error {:.2f}'.format(
  #       test_loss, 100 - 100. * test_acc))

  #   test_c_acc = test_c(net, test_data, base_c_path)
  #   print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))
  #   return

  # scheduler = torch.optim.lr_scheduler.LambdaLR(
  #     optimizer,
  #     lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
  #         step,
  #         args.epochs * len(train_loader),
  #         1,  # lr_lambda computes multiplicative factor
  #         1e-6 / args.learning_rate))

  # if not os.path.exists(args.save):
  #   os.makedirs(args.save)
  # if not os.path.isdir(args.save):
  #   raise Exception('%s is not a dir' % args.save)

  # log_path = os.path.join(args.save,
  #                         args.dataset + '_' + args.model + '_training_log.csv')
  # with open(log_path, 'w') as f:
  #   f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

  # best_acc = 0
  # print('Beginning training from epoch:', start_epoch + 1)
  # for epoch in range(start_epoch, args.epochs):
  #   begin_time = time.time()

  #   train_loss_ema = train(net, train_loader, optimizer, scheduler)
  #   test_loss, test_acc = test(net, test_loader)

  #   is_best = test_acc > best_acc
  #   best_acc = max(test_acc, best_acc)
  #   checkpoint = {
  #       'epoch': epoch,
  #       'dataset': args.dataset,
  #       'model': args.model,
  #       'state_dict': net.state_dict(),
  #       'best_acc': best_acc,
  #       'optimizer': optimizer.state_dict(),
  #   }

  #   save_path = os.path.join(args.save, 'checkpoint.pth.tar')
  #   torch.save(checkpoint, save_path)
  #   if is_best:
  #     shutil.copyfile(save_path, os.path.join(args.save, 'model_best.pth.tar'))

  #   with open(log_path, 'a') as f:
  #     f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
  #         (epoch + 1),
  #         time.time() - begin_time,
  #         train_loss_ema,
  #         test_loss,
  #         100 - 100. * test_acc,
  #     ))

  #   print(
  #       'Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} |'
  #       ' Test Error {4:.2f}'
  #       .format((epoch + 1), int(time.time() - begin_time), train_loss_ema,
  #               test_loss, 100 - 100. * test_acc))

  # test_c_acc = test_c(net, test_data, base_c_path)
  # print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))

  # with open(log_path, 'a') as f:
  #   f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' %
  #           (args.epochs + 1, 0, 0, 0, 100 - 100 * test_c_acc))


# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7' + args.name + '_'
                            + str(args.seed))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
else:
    print('==> Building model..')
    net = models.__dict__[args.model]()

if not os.path.isdir('results'):
    os.mkdir('results')
logname = ('results/log_' + net.__class__.__name__ + '_' + args.name + '_'
           + str(args.seed) + '.csv')

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    print('Using CUDA..')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                      weight_decay=args.decay)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                       args.alpha, use_cuda)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
        outputs = net(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        train_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
                        100.*correct/total, correct, total))
    return (train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total)


def test(net, test_loader):
  """Evaluate network on given dataset."""
  net.eval()
  total_loss = 0.
  total_correct = 0
  with torch.no_grad():
    for images, targets in test_loader:
      images, targets = images.cuda(), targets.cuda()
      logits = net(images)
      loss = F.cross_entropy(logits, targets)
      pred = logits.data.max(1)[1]
      total_loss += float(loss.data)
      total_correct += pred.eq(targets.data).sum().item()

  return total_loss / len(test_loader.dataset), total_correct / len(
      test_loader.dataset)


def test_c(net, test_data, base_path):
  """Evaluate network on given corrupted dataset."""
  corruption_accs = []
  for corruption in CORRUPTIONS:
    # Reference to original data is mutated
    test_data.data = np.load(base_path + corruption + '.npy')
    test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    test_loss, test_acc = test(net, test_loader)
    corruption_accs.append(test_acc)
    print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
        corruption, test_loss, 100 - 100. * test_acc))

  return np.mean(corruption_accs)

# def test(epoch):
#     global best_acc
#     net.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     for batch_idx, (inputs, targets) in enumerate(testloader):
#         if use_cuda:
#             inputs, targets = inputs.cuda(), targets.cuda()
#         inputs, targets = Variable(inputs, volatile=True), Variable(targets)
#         outputs = net(inputs)
#         loss = criterion(outputs, targets)

#         test_loss += loss.data
#         _, predicted = torch.max(outputs.data, 1)
#         total += targets.size(0)
#         correct += predicted.eq(targets.data).cpu().sum()

#         progress_bar(batch_idx, len(testloader),
#                      'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                      % (test_loss/(batch_idx+1), 100.*correct/total,
#                         correct, total))
#     acc = 100.*correct/total
#     if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
#         checkpoint(acc, epoch)
#     if acc > best_acc:
#         best_acc = acc
#     return (test_loss/batch_idx, 100.*correct/total)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7' + args.name + '_'
               + str(args.seed))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                            'test loss', 'test acc'])

for epoch in range(start_epoch, args.epoch):
    train_loss, reg_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    adjust_learning_rate(optimizer, epoch)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss,
                            test_acc]

test_c(net, testset, './data/cifar/CIFAR-10-C/')
