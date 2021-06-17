# -*- coding: utf-8 -*-
import pandas as pd
import torch
import os
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score
from collections import OrderedDict, namedtuple, defaultdict
import random
import multiprocessing
import heapq
import time
import sys
import pickle
from prettytable import PrettyTable
sys.path.append('../')
import util.metrics as metrics

class MF(nn.Module):

    def __init__(self, n_user, n_item,
                 embedding_size=4, l2_reg_embedding=0.00001, init_std=0.0001,seed=1024,):

        super(MF, self).__init__()

        self.model_name = 'mf'
        self.n_user = n_user + 1
        self.n_item = n_item + 1
        self.embedding_size = embedding_size

        self.l2_reg_embedding = l2_reg_embedding
        self.embedding_dict = nn.ModuleDict({
            'user_emb': self.create_embedding_matrix(self.n_user, embedding_size),
            'item_emb': self.create_embedding_matrix(self.n_item, embedding_size),
        })

    def create_embedding_matrix(self, vocabulary_size, embedding_size, init_std=0.0001, sparse=False, ):
        embedding = nn.Embedding(vocabulary_size, embedding_size, sparse=sparse)
        nn.init.normal_(embedding.weight, mean=0, std=init_std)
        return embedding

    def computeL2LOSS(self, input_dict):
        users, pos_items, neg_items = input_dict['users'], input_dict['pos_items'], input_dict['neg_items']

        user_vector = self.embedding_dict['user_emb'](users)
        pos_items_vector = self.embedding_dict['item_emb'](pos_items)
        neg_items_vector = self.embedding_dict['item_emb'](neg_items)

        emb_loss = torch.norm(user_vector) ** 2 + \
                   torch.norm(pos_items_vector) ** 2 + torch.norm(neg_items_vector) ** 2
        return emb_loss

    def forward(self, input_dict):
        '''

        :param input_dict:
        :return:   rui, ruj
        '''
        users, pos_items, neg_items = input_dict['users'], input_dict['pos_items'], input_dict['neg_items']

        user_vector = self.embedding_dict['user_emb'](users)
        pos_items_vector = self.embedding_dict['item_emb'](pos_items)
        neg_items_vector = self.embedding_dict['item_emb'](neg_items)

        rui = torch.sum(torch.mul(user_vector, pos_items_vector), dim=-1, keepdim=True)
        ruj = torch.sum(torch.mul(user_vector, neg_items_vector), dim=-1, keepdim=True)

        mf_loss = self.bpr_loss(rui,ruj)
        emb_loss = self.computeL2LOSS(input_dict)

        emb_loss = self.l2_reg_embedding * emb_loss
        return mf_loss, emb_loss

    def rating(self, user_batch, all_item):
        user_vector = self.embedding_dict['user_emb'](user_batch)
        items_vector = self.embedding_dict['item_emb'](all_item)
        return torch.mm(user_vector, items_vector.t())

    def bpr_loss(self, p_score, n_score):
        return -torch.mean(torch.log(torch.sigmoid(p_score - n_score)))