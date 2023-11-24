#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from re import T
from select import select
import numpy as np
import math
from utils.fsv import multi_fsv_t
import logging

class Attacker():
    def __init__(self, alpha:float, args, logger) -> None:
        # alpha为攻击者控制节点的比率, 至少控制一个节点
        self.num_att = max(int(alpha * args.num_users), 1)
        # 目前设定为, 攻击者所掌握的节点不会变化.
        self.att_users = np.random.choice(range(args.num_users), self.num_att, replace=False)
        self.sharpleyDict = dict(zip(list(self.att_users), [0] * self.num_att))
        self.logger = logger
        self.logger.info("att_users: {}".format(self.att_users))
    
    # delta为攻击掉线数量占控制节点的比率.
    def dropAttack(self, idxs_users:np.array, delta:float) -> np.array:
        # print("att_users ", self.att_users)
        # 掉线攻击方案0: 不掉线
        if delta == 0:
            dropped_users = idxs_users
        # 掉线攻击方案1: 全掉线
        elif delta == 1:
            if (set(idxs_users) - set(self.att_users)):
                drop_users = self.att_users
                dropped_users = np.array(list(set(idxs_users) - set(drop_users)))
            else:# 如果一口气全掉完了
                print(idxs_users[0])
                dropped_users = np.array([idxs_users[0]])



        # 掉线攻击方案2: 随机部分掉线
        elif delta > 0 and delta < 1:
            # 随机的从 攻击者掌控的节点 ∩ 被选中的节点
            and_set = set(idxs_users) & set(self.att_users)
            # 这个集合中 选择num_drop个节点进行掉线.
            num_drop = math.floor(delta * len(and_set))
            drop_users = np.random.choice(list(and_set), num_drop, replace=False)
            dropped_users = np.array(list(set(idxs_users) - set(drop_users)))

        # 掉线攻击方案3: 使用夏普利值 delta为负时启动方案3，delta的绝对值为掉线比率。
        elif delta < 0 and delta > -1:
            delta = -delta
            # 随机的从 攻击者掌控的节点 ∩ 被选中的节点
            and_set = set(idxs_users) & set(self.att_users)
            # 这个集合中 选择num_drop个节点进行掉线.
            num_drop = math.floor(delta * len(and_set))
            # drop_users = np.random.choice(list(and_set), num_drop, replace=False)
            and_set_with_sharpley = []
            for (key,value) in self.sharpleyDict.items():
                if key in and_set:
                    and_set_with_sharpley.append((key,value))

            sort_by_sharpley = sorted(and_set_with_sharpley, key=lambda x: x[1], reverse=True)
            self.logger.info("\t\tsort_by_sharpley: {}".format(sort_by_sharpley))
            drop_users = []
            for i in range(num_drop):
                drop_users.append(sort_by_sharpley[i][0])
            
            dropped_users = np.array(list(set(idxs_users) - set(drop_users)))

        else:
            self.logger.info("delta输入错误！")
            

        # print("\t\tdrop_users: ", drop_users)
        return dropped_users
    
    def calSharpley(self, select_w_locals, select_users, dataset_test, args, net_glob):
        con_att_users = []
        con_w_locals = []
        select_users = list(select_users)
        for i in range(len(select_users)):
            if select_users[i] in self.att_users:
                con_att_users.append(select_users[i])
                con_w_locals.append(select_w_locals[i])
        if not len(con_w_locals) == 0:
            temp_s = multi_fsv_t(con_w_locals, dataset_test, args, net_glob)
            for i in range(len(temp_s)):
                self.sharpleyDict[con_att_users[i]] += temp_s[i]
       