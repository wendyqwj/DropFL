#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
from http.client import TEMPORARY_REDIRECT
from tempfile import tempdir
import torch
from torch import nn
from collections import defaultdict
import numpy as np

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def Krum(users_grads, users_count, corrupted_count, distances=None, return_index=False):
    if not return_index:
        assert users_count >= 2*corrupted_count + 1,('users_count>=2*corrupted_count + 3', users_count, corrupted_count)
    # 未被控制的节点
    non_malicious_count = int(users_count - corrupted_count)
    minimal_error = 1e20
    minimal_error_index = -1

    if distances is None:
        # i到j的欧式距离矩阵
        distances = defaultdict(dict)
        for i in range(len(users_grads)):
            for j in range(i):
                temp_i = torch.FloatTensor([])
                temp_j = torch.FloatTensor([])
                for k in users_grads[i].keys():
                    for l in range(len(users_grads[i][k])):
                        temp_i = torch.cat((temp_i, users_grads[i][k][l]),0)
                        temp_j = torch.cat((temp_j, users_grads[j][k][l]),0)
                distances[i][j] = distances[j][i] = torch.sqrt(torch.sum((temp_i - temp_j) ** 2, dim=2))
    for user in distances.keys():
        errors = sorted(distances[user].values())
        current_error = sum(errors[:non_malicious_count])
        if current_error < minimal_error:
            minimal_error = current_error
            minimal_error_index = user

    if return_index:
        return minimal_error_index
    else:
        return users_grads[minimal_error_index]