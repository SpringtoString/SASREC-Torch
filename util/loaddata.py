import numpy as np
import pandas as pd
import random as rd
import scipy.sparse as sp
from time import time
from collections import defaultdict


class Data(object):
    def __init__(self, path, max_len=10):
        '''
        '''
        self.path = path
        self.max_len = max_len

        train_file = path
        # get number of users and items
        self.n_users, self.n_items = 0, 0

        user2items= defaultdict(list)

        self.n_train = 0
        with open(train_file, 'r') as f:
            for line in f:
                u, i = line.rstrip().split(' ')
                u = int(u)
                i = int(i)
                self.n_train =  self.n_train + 1
                self.n_users = max(u, self.n_users)
                self.n_items = max(i, self.n_items)
                user2items[u].append(i)

        self.n_train = self.n_train - 2*self.n_users

        # 按照时间顺序划分训练集，验证集，测试集
        self.user_train, self.user_valid, self.user_test = defaultdict(list), defaultdict(list), defaultdict(list)

        for u,items in user2items.items():
            nfeedback = len(items)
            if nfeedback < 3:
                self.user_train[u] = items
            else:
                self.user_train[u] = items[:-2]
                self.user_valid[u].append(items[-2])
                self.user_test[u].append(items[-1])

        # 处理成padding序列化形式：    TODO 使用切片操作优化处理程序
        self.user2seq={}
        self.user2positem = {}       # 正标签 训练时会用到
        self.user2valseq, self.user2testseq = {}, {}    # 验证，测试时的序列

        for u in self.user_train.keys():
            seq = np.zeros([max_len], dtype=np.int32)
            pos = np.zeros([max_len], dtype=np.int32)

            val_seq, test_seq = np.zeros([max_len], dtype=np.int32), np.zeros([max_len], dtype=np.int32)
            items = self.user_train[u]

            nxt = items[-1]
            idx, val_idx, test_idx = max_len - 1, max_len - 1, max_len - 1
            # val_seq, test_seq 末尾的item要先处理好
            test_seq[test_idx] = self.user_valid[u][0]
            test_idx = test_idx -1

            val_seq[val_idx] = items[-1]
            test_seq[test_idx] = items[-1]
            val_idx = val_idx - 1
            test_idx = test_idx - 1

            for i in reversed(items[:-1]):
                if val_idx>=0:
                    val_seq[val_idx] = i
                    val_idx = val_idx - 1
                if test_idx>=0:
                    test_seq[test_idx] = i
                    test_idx = test_idx -1
                seq[idx] = i
                pos[idx] = nxt
                nxt = i
                idx -= 1
                if idx == -1: break
            self.user2seq[u] = seq
            self.user2positem[u] = pos
            self.user2valseq[u] = val_seq
            self.user2testseq[u] = test_seq


        self.exist_users = [i for i in range(1, self.n_users+1)]     # user 从1开始编号
        self.all_item = [i for i in range(0, self.n_items+1)]        # item从1开始编号，0作为占位符

        self.print_statistics()

    def sample(self,batch_size=256):
        if  batch_size <= self.n_users:
            users = rd.sample(self.exist_users, batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(batch_size)]

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = self.user_train[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=1, high=self.n_items+1,size=1)[0]
                if neg_id not in self.user_train[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def seq_sample(self,batch_size=256):
        '''
        :return: [user id], [postiveitem id] [negativeitem id]
        '''
        if  batch_size <= self.n_users:
            users = rd.sample(self.exist_users, batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(batch_size)]

        def random_neq(l, r, s):
            t = np.random.randint(l, r)
            while t in s:
                t = np.random.randint(l, r)
            return t

        def sample_neg_items_for_u(u):
            # sample num neg items for u-th user
            neg_items = np.zeros([self.max_len], dtype=np.int32)
            ts = set(self.user_train[u])  # 负样本要取 user_train交互以外的

            for idx in range(self.max_len-1, -1, -1):
                neg_items[idx] = random_neq(1, self.n_items + 1, ts)

                if self.user2seq[u][idx]==0: break

            return neg_items

        seqs, pos_items, neg_items = [], [], []
        for u in users:
            seqs.append(self.user2seq[u])
            pos_items.append(self.user2positem[u])
            neg_items.append(sample_neg_items_for_u(u))

        return users, seqs, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('train instance=%d' %(self.n_train))

    def get_test_sample(self, user_batch, mode='val'):
        seqs, next_items = [], []
        if mode=='val':
            for u in user_batch:
                seqs.append(self.user2valseq[u])
                next_items.append(self.user_valid[u])
        elif mode=='test':
            for u in user_batch:
                seqs.append(self.user2testseq[u])
                next_items.append(self.user_test[u])

        return seqs, next_items

