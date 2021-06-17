import os
import time
import torch
import torch.nn as nn
import argparse
import numpy as np

import warnings
warnings.filterwarnings('ignore')
from time import time
import random

from models import sas
from trainers import basetrainer
from trainers.basetrainer import SASTrainer
from models.sas import SAS
from util.loaddata import Data
from util.utils import set_seed , get_device
import multiprocessing
import heapq
import util.metrics as metrics

# python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', required=True)
# parser.add_argument('--train_dir', required=True)
# parser.add_argument('--batch_size', default=128, type=int)
# parser.add_argument('--lr', default=0.001, type=float)
# parser.add_argument('--maxlen', default=50, type=int)
# parser.add_argument('--hidden_units', default=50, type=int)
# parser.add_argument('--num_blocks', default=2, type=int)
# parser.add_argument('--num_epochs', default=201, type=int)
# parser.add_argument('--num_heads', default=1, type=int)
# parser.add_argument('--dropout_rate', default=0.5, type=float)
# parser.add_argument('--l2_emb', default=0.0, type=float)
# parser.add_argument('--device', default='cpu', type=str)
# parser.add_argument('--inference_only', default=False, type=str2bool)
# parser.add_argument('--state_dict_path', default=None, type=str)
#
# args = parser.parse_args()

if __name__ == '__main__':
    set_seed()
    device = get_device(use_cuda=True, gpu_id=0)

    filepath = 'Data/ml-1m.txt'
    data_generator = Data(path=filepath, max_len=200,)  # mode=1 设置为测试集模式
    basetrainer.data_generator = data_generator           # batch_size=args.batch_size
    print("read complete")

    model = SAS(data_generator.n_users, data_generator.n_items,max_len=200, embedding_size = 10, l2_reg_embedding=0.025)
    trainer = SASTrainer(model, lr=0.001, batch_size=128, epochs=100, verbose=5, save_round=100, early_stop=False, device=device)
    trainer.fit()




