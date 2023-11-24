#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
import math
import random
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def noniid_pro(dataset, num_users, P):
    """
    Sample non-I.I.D client data from MNIST dataset based on probability P
    :param dataset:
    :param num_users: The number of users, recommended to be an integer multiple of the number of labels
    :param P: The probability that a label is assigned to the correct group
    :return: a dict for users that contains the corresponding index of imgaes
    NOTE: num_users should NOT be less than the number of labels
    """
    print(f"noniid: num_users={num_users} P={P}")
    num_imgs = len(dataset)
    LABEL_NUM=len(set(dataset.targets.tolist())) if not isinstance(dataset.targets,list) else len(set(dataset.targets))
    user_group_size=math.ceil(num_users/LABEL_NUM)

    dict_users = dict()
    img_idxs = np.arange(num_imgs)
    labels = np.array(dataset.targets)
    rand_factors = [random.random() for _ in range(num_imgs)]
    
    # divide to users
    groups= [list() for _ in range(LABEL_NUM)]
    for id in img_idxs:
        if rand_factors[id]<=P:
            groups[labels[id]].append(id)
        else:
            rand_gp = random.randint(0,LABEL_NUM-1)
            while rand_gp==labels[id]:
                rand_gp = random.randint(0,LABEL_NUM-1)
            groups[rand_gp].append(id)
    shards_size=[int(len(groups[group_id])/user_group_size) for group_id in range(LABEL_NUM)]

    group_id=0
    group_begin=0
    for user_id in range(num_users):
        img_ids=groups[group_id][group_begin:group_begin+shards_size[group_id]]
        group_begin+=shards_size[group_id]
        group_remain=len(groups[group_id])-group_begin
        if group_remain<shards_size[group_id] or user_id == num_users-1: # the last one in the group get all the remain images
            img_ids+=groups[group_id][group_begin:]
            group_id+=1
            group_begin=0
        dict_users[user_id]=np.array(img_ids)
    
    return dict_users



if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
