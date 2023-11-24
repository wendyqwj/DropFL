import numpy
import copy
import torch

from torch.utils.data import DataLoader
from utils.options import args_parser
from math import log2
from models.Fed import FedAvg
from models.test import test_accuracy

args = args_parser()

_testloader = None
_POOL = None
_QUE = None
_MAX_WORKERS = args.MAX_WORKERS  # 进程太多显存会炸

# fsv parameters
_r = None
_epsilon = None
_deta = None
_coe = None


# def fsv_t(I, dataset, args, net, v=test_accuracy):
#     """ 
#     Calculate the sharpley value for round t
#     I is the controlled client set for round t
#     v is the value function, which receives a client list as a parameter
#     """
#     global _r
#     global _epsilon
#     global _deta
#     global _coe
#     if _r is None or _epsilon is None or _deta is None:
#         _r = 1
#         _epsilon = 0.1
#         _deta = 0.1
#     if _coe is None:
#         _coe = 2*_r**2/_epsilon**2
#     m = len(I)
#     T = int(_coe*log2(2*m/_deta))

#     global _testloader
#     if _testloader is None:
#         _testloader = DataLoader(dataset, batch_size=args.bs)

#     prev = v(net, _testloader, args)
#     s = [0]*m
#     for _ in range(T):
#         rand_I = numpy.random.permutation(range(m))
#         for i in range(m):
#             w_curr = FedAvg([I[rand_I[idx]] for idx in range(i+1)])
#             net.load_state_dict(w_curr)
#             curr = v(net, _testloader, args)
#             s[rand_I[i]] += curr-prev
#             prev = curr
#     return s


def fsv_core(I, net, v, args, prev, testloader, que):
    net=copy.deepcopy(net)
    m = len(I)
    rand_I = numpy.random.permutation(range(m))
    s = [0]*m
    for i in range(m):
        w_curr = FedAvg([I[rand_I[idx]] for idx in range(i+1)])
        net.load_state_dict(w_curr)
        curr = v(net, testloader, args)
        s[rand_I[i]] += curr-prev
        prev = curr
    que.put(s)
    if args.verbose:
        print(f"fsv_core result {s}")


def multi_fsv_t(I, dataset, args, net, v=test_accuracy):
    global _POOL
    global _QUE
    global _MAX_WORKERS
    if _POOL is None:
        ctx = torch.multiprocessing.get_context("spawn")
        _POOL = ctx.Pool(_MAX_WORKERS)
        _QUE = ctx.Manager().Queue()
    global _r
    global _epsilon
    global _deta
    global _coe
    if _r is None or _epsilon is None or _deta is None:
        _r = 1
        _epsilon = 0.2
        _deta = 0.2
    if _coe is None:
        _coe = 2*_r**2/_epsilon**2
    m = len(I)
    # print('m', m)
    T = int(_coe*log2(2*m/_deta))
    # print('T', T)
    # T = 1     # 调试时加快

    global _testloader
    if _testloader is None:
        _testloader = DataLoader(dataset, batch_size=args.bs)

    prev = v(net, _testloader, args)
    s = [0]*m

    for _ in range(T):
        _POOL.apply_async(fsv_core, (I, net,
                          v, args, prev, _testloader, _QUE))

    for _ in range(T):
        iter_s = _QUE.get()
        for i in range(m):
            s[i] += iter_s[i]
    for i in range(m):
        s[i] = s[i] / T
    return s
