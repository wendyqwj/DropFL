#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from time import time


def test_img(net_g, dataset, args):
    net_g.eval() # 要加, 不要删, 3.7晚找了3小时
    start = time()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(dataset, batch_size=args.bs)
    for data, target in data_loader:
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\nTime cost: {}\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy, time()-start))
    return accuracy, test_loss

def test_accuracy(net, data_loader, args):
    start = time()
    net.to(args.device)
    with torch.no_grad():  # when in test stage, no grad
        correct = 0
        total = 0
        for (imgs, labels) in data_loader:
            imgs = imgs.to(args.device)
            labels = labels.to(args.device)
            out = net(imgs)
            _, pre = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (pre == labels).sum().item()
    accuracy = 100.00*correct / total
    if args.verbose:
        print('\nAccuracy: {}/{} {:.2f}%\nTime cost: {}\n'.format(correct,
              total, accuracy, time()-start))
    return accuracy
