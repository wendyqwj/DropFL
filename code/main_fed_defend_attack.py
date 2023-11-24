#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import logging
from datetime import datetime
from models.drop import Attacker
from models.test import test_img
from models.Fed import FedAvg
from models.Nets import MLP, CNNMnist, CNNCifar, CNNCifar100
from models.Update import LocalUpdate
from utils.options import args_parser
import torch
from torchvision import datasets, transforms
import numpy as np
import copy
import matplotlib.pyplot as plt
from utils.sampling import mnist_iid, cifar_iid, noniid_pro
from utils.select_users import random_select, health_select
from utils.detection import detect, detratio_vs_num
from cProfile import label
from re import A
import matplotlib
matplotlib.use('Agg')


class FedTrain():
    def __init__(self, args, img_size, delta=0) -> None:
        # build model
        if args.model == 'cnn' and args.dataset == 'cifar':
            self.net_glob = CNNCifar(args=args).to(args.device)
        elif args.model == 'cnn' and args.dataset == 'mnist':
            self.net_glob = CNNMnist(args=args).to(args.device)
        elif args.model == 'cnn' and args.dataset == "cifar100":
            self.net_glob = CNNCifar100().to(args.device)
        elif args.model == 'mlp':
            self.len_in = 1
            for x in img_size:
                self.len_in *= x
            self.net_glob = MLP(dim_in=self.len_in, dim_hidden=200,
                                dim_out=args.num_classes).to(args.device)
        else:
            exit('Error: unrecognized model')

        self.net_glob.train()
        self.w_glob = self.net_glob.state_dict()

        # training
        self.loss_train = []
        self.accuracy_test = []
        self.cv_loss, self.cv_acc = [], []
        self.val_loss_pre, self.counter = 0, 0
        self.net_best = None
        self.best_loss = None
        self.val_acc_list, self.net_list = [], []
        # 决定了该训练过程被攻击的类型
        self.delta = delta

        if args.all_clients:
            logger.info("Aggregation over all clients")
            self.w_locals = [self.w_glob for i in range(args.num_users)]

    # 一轮训练全过程，通过delta来控制掉线情况，返回这轮参与训练的users
    def oneEpoch(self, args, idxs_users):

        dped_users = attacker.dropAttack(idxs_users, self.delta)  # 对照组

        self.loss_locals = []
        if not args.all_clients:
            self.w_locals = []
        for idx in dped_users:
            local = LocalUpdate(
                args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(
                self.net_glob).to(args.device))
            if args.all_clients:
                self.w_locals[idx] = copy.deepcopy(w)
            else:
                self.w_locals.append(copy.deepcopy(w))
            self.loss_locals.append(copy.deepcopy(loss))

        # update global weights
        if len(self.w_locals):
            self.w_glob = FedAvg(self.w_locals)

        # # cal sharpley
        if -1 < self.delta < 0:
            attacker.calSharpley(self.w_locals, dped_users,
                                 dataset_test, args, self.net_glob)

        # copy weight to net_glob
        self.net_glob.load_state_dict(self.w_glob)

        # print loss
        if len(self.loss_locals):
            loss_avg = sum(self.loss_locals) / len(self.loss_locals)
        else:
            loss_avg = -1  # error state
        self.loss_train.append(loss_avg)

        # print accuracy
        # print("delta = {}, self.net_glob: {}".format(self.delta, self.net_glob.eval()))
        acc_test, _ = test_img(self.net_glob, dataset_test, args)
        self.accuracy_test.append(float(acc_test))
        logger.info("\tdelta:{:.1f} \t Average loss {:.3f} \t Testing accuracy: {:.2f} \tdped_users:{}".format(
            self.delta, loss_avg, acc_test, dped_users))
        return dped_users


if __name__ == '__main__':
    startTime = datetime.now()
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(
        args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if args.defend and args.detect:
        exit('Error: detect mode and defend mode are mutually exclusive')
    # 输出到文件
    if args.defend:
        head = 'defend'
    elif args.detect:
        head = 'detect'
    else:
        head = 'no-defend'
    fh = logging.FileHandler('./log/{}_log_{}_{}_epo{}_n{}_C{}_a{}_d{}_iid{}_P_{}_{}.log'.format(head, args.dataset, args.model,
                                                                                                 args.epochs, args.num_users, args.frac, args.alpha, args.delta, args.iid, args.P, datetime.now().strftime('%Y-%m-%d-%H-%M-%S')), mode='w')
    fh.setLevel(logging.DEBUG)
    # 输出到终端
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)
    # logging.basicConfig(level=logging.DEBUG,
    #                 format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')  # logging.basicConfig函数对日志的输出格式及方式做相关配置

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(
            '../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(
            '../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = noniid_pro(dataset_train, args.num_users, args.P)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10(
            '../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10(
            '../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = noniid_pro(dataset_train, args.num_users, args.P)
    elif args.dataset == 'cifar100':
        trans_cifar = transforms.Compose([
            transforms.Resize(32),  # 将图像转化为32 * 32
            transforms.RandomHorizontalFlip(p=0.75),  # 有0.75的几率随机旋转
            # transforms.ColorJitter(brightness=1, contrast=2, saturation=3, hue=0),  # 给图像增加一些随机的光照
            transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
            transforms.Normalize([0.485, 0.456, 0.406], [
                                 0.229, 0.224, 0.225])  # 归一化
        ])
        dataset_train = datasets.CIFAR100(
            '../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR100(
            '../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = noniid_pro(dataset_train, args.num_users, args.P)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # fed0 = FedTrain(args, img_size, 0)
    # fed1 = FedTrain(args, img_size, 1)
    # fed2 = FedTrain(args, img_size, args.delta)
    fed3 = FedTrain(args, img_size, -args.delta)
    # fed4 = FedTrain(args, img_size, 0.5)
    # fed5 = FedTrain(args, img_size, -0.5)
    # fedList = [fed0, fed1, fed2, fed3, fed4, fed5]
    fedList = [fed3]

    # 初始化一个攻击者
    attacker = Attacker(args.alpha, args, logger)
    # 初始化健康值dict
    health_dict = dict(zip(range(args.num_users), [1]*args.num_users))
    detec_rates = []

    for iter in range(args.epochs):
        # 客户端参与度C, 这里用原程序的--frac控制, 默认为0.1
        m = max(int(args.frac * args.num_users), 1)
        if args.defend:
            idxs_users = health_select(health_dict, m)
        else:
            idxs_users = random_select(health_dict, m)
        logger.info("\n===================Round {:3d}====================\t\t\t\tidxs_users:{}".format(
            iter, idxs_users))

        for fedX in fedList:
            participants = fedX.oneEpoch(args, idxs_users)
            for id in participants:
                health_dict[id] += 1
            for id in set(idxs_users)-set(participants):
                health_dict[id] -= 1
            if args.defend:
                logger.info(f"health value: {health_dict}")
            if args.detect:
                sorted_list = sorted(health_dict.items(),
                                     key=lambda x: x[1], reverse=False)
                sorted_ids = [x[0] for x in sorted_list]
                detec_rates.append(
                    detect(list(attacker.att_users), sorted_ids))
                logger.info(
                    f"detection rate: {detec_rates[-1]}")

        endTime = datetime.now()
        logger.info('\nRunning time: %s Seconds' % (endTime-startTime))
        if (iter+1) % args.record_round == 0:
            logger.debug(
                "\n===================DATA Round 0-{}====================".format(iter))
            logger.debug("Accuracy:")
            for fedX in fedList:
                logger.debug('{},'.format(fedX.accuracy_test))
            logger.debug("Loss:")
            for fedX in fedList:
                logger.debug('{},'.format(fedX.loss_train))
            logger.debug(
                "===================END Round 0-{}====================".format(iter))

#    # plot loss curve
#     plt.figure()
#     plt.ylim(60,100)
#     for fedX in fedList:
#         plt.plot(range(len(fedX.accuracy_test)), fedX.accuracy_test,
#                         label="delta={:.2f}".format(fedX.delta))
#     plt.ylabel('test_accuracy')
#     plt.legend()
#     plt.savefig('./save/accu_{}_{}_epo{}_n{}_C{}_a{}_d{}_iid{}_{}.png'.format(args.dataset, args.model,
#                  args.epochs, args.num_users, args.frac,args.alpha, args.delta, args.iid, datetime.now().strftime('%Y-%m-%d-%H:%M:%S')))

    logger.info("\n========TRAIN END========\n")
    if args.detect:
        sorted_list = sorted(health_dict.items(),
                             key=lambda x: x[1], reverse=False)
        sorted_ids = [x[0] for x in sorted_list]
        logger.info(
            f"detection rates: {detec_rates}")
        logger.info(
            f"sorted_ids: {sorted_ids}\natt_users: {attacker.att_users}")
        logger.info(
            f"detection ratio vs num: {detratio_vs_num(sorted_ids,attacker.att_users)}")

    # testing
    for fedX in fedList:
        logger.info("\ndelta = {}".format(fedX.delta))
        fedX.net_glob.eval()
        acc_train, loss_train = test_img(fedX.net_glob, dataset_train, args)
        acc_test, loss_test = test_img(fedX.net_glob, dataset_test, args)
        logger.info("Loss train: {:.2f}".format(loss_train))
        logger.info("Testing accuracy: {:.2f}".format(acc_test))
        logger.info(fedX.accuracy_test)
