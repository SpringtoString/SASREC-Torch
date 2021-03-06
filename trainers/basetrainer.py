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
import logging
import datetime
import sys
import setproctitle
from prettytable import PrettyTable

# import  pyximport
# pyximport.install()
sys.path.append('../')
import util.metrics as metrics
from evaluator.cpp.cpp_evaluator import CPPEvaluator
from  util.utils import EarlyStopManager

data_generator = 0 # Data(path=filepath) 主程序中传入

def get_auc(item_score, user_pos_test):
    '''

    :param item_score: dict,待选item的预测评分
    :param user_pos_test: user 测试集中真实交互的item
    :return: auc
    '''
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()

    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def test_one_user(x, Ks = [5, 10]):
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]
    ITEM_NUM = data_generator.n_items

    # user u's items in the training set
    try:
        training_items = data_generator.train_items[u]  # user 已交互的item
    except Exception:
        training_items = []
    # user u's items in the test set
    user_pos_test = data_generator.test_set[u]  # 测试集中真实的item

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))  # 待选的item

    def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
        '''

        :param user_pos_test: user 测试集中真实交互的item
        :param test_items:    待选item
        :param rating:        user的所有预测评分
        :param Ks:            TOP-K
        :return:
        '''
        item_score = {}
        for i in test_items:
            item_score[i] = rating[i]

        K_max = max(Ks)
        K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

        r = []
        for i in K_max_item_score:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        # auc = get_auc(item_score, user_pos_test)
        auc = 0
        return r, auc

    def get_performance(user_pos_test, r, auc, Ks):
        '''

        :param user_pos_test:    user 测试集中真实交互的item
        :param r:                r = [1,0,1] 表示预测TOP-K是否命中
        :param auc:              auc =0 标量
        :param Ks:               TOP-K
        :return:
        '''
        precision, recall, ndcg, hit_ratio, MAP = [], [], [], [], []

        for K in Ks:
            precision.append(metrics.precision_at_k(r, K))
            recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
            ndcg.append(metrics.ndcg_at_k(r, K))
            hit_ratio.append(metrics.hit_at_k(r, K))
            MAP.append(metrics.AP_at_k(r, K, len(user_pos_test)))

        return {'recall': np.array(recall), 'precision': np.array(precision), 'ndcg': np.array(ndcg),
                'hit_ratio': np.array(hit_ratio), 'MAP': np.array(MAP), 'auc': auc}

    # if args.test_flag == 'part':
    #     r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    # else:

    r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


class BaseTrainer(object):

    def __init__(self, model, lr=0.001, batch_size=500, epochs=15, verbose=5, save_round=200, Ks = [5, 10], early_stop=False,patience=10,
                 device='cpu'):

        self.model = model
        self.model.to(device)
        self.model.device = device
        self.name = 'basetrainer_' + self.model.model_name

        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.verbose = verbose
        self.save_round = save_round
        self.Ks = Ks
        self.early_stop = early_stop
        self.patience = patience
        self.optimizer = self.get_optimizer()
        self.data_generator = data_generator
        self.set_eval_list()
        self.device = device

    def get_optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0)
        return optimizer

    def set_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)

        log_dir = 'logs'
        log_dir = os.path.join(log_dir, data_generator.path.split('/')[-1])
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # log file handler
        log_path = os.path.join(log_dir,'log_%s.txt'%(self.name))
        handler = logging.FileHandler(log_path, mode='w')
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        # console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(console)
        self.logger = logger

    def set_process_name(self):
        date_str = datetime.datetime.now().strftime('%d%H%M%S')
        setproctitle.setproctitle(self.name+ date_str + '@SBZHAZHA')

    def set_esm(self):
        self.esm = EarlyStopManager(patience=self.patience)

    def print_device_info(self):
        # print("train on ", self.device)
        self.logger.info("train on %s" % (self.device))

    def get_input_data(self):
        users, pos_items, neg_items = data_generator.sample(self.batch_size)

        users = torch.from_numpy(np.array(users)).to(self.device).long()
        pos_items = torch.from_numpy(np.array(pos_items)).to(self.device).long()
        neg_items = torch.from_numpy(np.array(neg_items)).to(self.device).long()

        input_dict = {'users': users, 'pos_items': pos_items, 'neg_items': neg_items}
        return input_dict

    def get_n_batch(self):
        # 显示 一次epoch需要几个step
        sample_num = self.data_generator.n_train
        n_batch = (sample_num - 1) // self.batch_size + 1

        # print("Train on {0} samples,  {1} steps per epoch".format(sample_num, n_batch))
        self.logger.info("Train on {0} samples,  {1} steps per epoch".format(sample_num, n_batch))
        return n_batch

    def set_eval_list(self):
        self.prec_list, self.rec_list, self.ndcg_list, self.hr_list, self.ap_list = [], [], [], [], []

    def log_result(self,epoch, result, eval_time):
        eval_res = PrettyTable()
        eval_res.field_names = ['epoch', 'time', 'precision', 'recall', 'ndcg', 'hit_ratio', 'MAP']
        eval_res.add_row([epoch, eval_time, result['precision'], result['recall'], result['ndcg'], result['hit_ratio'],
                          result['MAP']])
        # print(eval_res)
        self.logger.info(eval_res)
        # print(" ")
        self.logger.info(" ")

        self.prec_list.append(result['precision'][0])
        self.rec_list.append(result['recall'][0])
        self.ndcg_list.append(result['ndcg'][0])
        self.hr_list.append(result['hit_ratio'][0])
        self.ap_list.append(result['MAP'][0])

    def save_model(self, epoch):
        save_dir = 'Save'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        model_path = os.path.join('Save', 'checkpoint-%s' % (self.model.model_name))
        torch.save(checkpoint, model_path)

    def load_model(self):
        model_path = os.path.join('Save', 'checkpoint-%s' % (self.model.model_name))
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model'])
        print('load model complete!')

    def save_result_csv(self):
        res_dir = 'RES'
        res_dir = os.path.join(res_dir, data_generator.path.split('/')[-1])
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        csv_name = os.path.join(res_dir,'%s.csv' % (self.name))

        df = pd.DataFrame(
            {'precision': self.prec_list, 'recall': self.rec_list, 'ndcg': self.ndcg_list, 'hr': self.hr_list, 'MAP': self.ap_list})
        df.to_csv(csv_name, index=False)

    def fit(self):
        self.set_logger()
        self.set_process_name()
        self.print_device_info()

        if self.early_stop:
            self.set_esm()

        n_batch = self.get_n_batch()

        should_stop = False
        for epoch in range(1, self.epochs+1):
            total_loss, total_mf_loss, total_emb_loss = 0.0, 0.0, 0.0
            with torch.autograd.set_detect_anomaly(True):
                start_time = time.time()
                for index in range(n_batch):
                    input_dict = self.get_input_data()
                    mf_loss, emb_loss = self.model(input_dict)

                    self.optimizer.zero_grad()

                    reg_emb_loss = emb_loss/self.batch_size
                    loss = mf_loss + reg_emb_loss

                    loss.backward(retain_graph=True)
                    self.optimizer.step()

                    total_mf_loss = total_mf_loss + mf_loss.item()
                    total_emb_loss = total_emb_loss + reg_emb_loss.item()
                    total_loss = total_loss + loss.item()

                epoch_time = time.time() - start_time
                loss_info = 'epoch %d %.2fs train loss is [%.4f = %.4f + %.4f] ' % (epoch, epoch_time,
                            total_loss / n_batch, total_mf_loss/n_batch, total_emb_loss/n_batch)
                # print(loss_info)
                self.logger.info(loss_info)

            if epoch==1 or epoch % self.verbose == 0 or epoch == self.epochs :
                start_time = time.time()
                result = self.test_cpp(batch_size=2*self.batch_size, Ks=self.Ks)
                eval_time = time.time() - start_time
                self.log_result(epoch, result, eval_time)
                if self.early_stop:
                    should_stop = self.esm.step(result['recall'][0])
            # 保存模型
            if (self.save_round != -1 and epoch % self.save_round == 0) or epoch==self.epochs or should_stop:
                self.save_model(epoch)
            # early stop
            if should_stop:
                print("---early stop---")
                break

        # 保存csv结果
        self.save_result_csv()

    def test(self, batch_size=256, Ks = [5, 10]):
        cores = multiprocessing.cpu_count() // 2
        pool = multiprocessing.Pool(cores)

        # data_generator = self.data_generator
        result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
                  'hit_ratio': np.zeros(len(Ks)), 'MAP': np.zeros(len(Ks)), 'auc': 0.}

        u_batch_size = batch_size
        i_batch_size = batch_size

        test_users = list(data_generator.test_set.keys())
        n_test_users = len(test_users)
        n_user_batchs = (n_test_users - 1) // u_batch_size + 1

        count = 0
        # with torch.no_grad():
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            # end 这里需要完善
            end = (u_batch_id + 1) * u_batch_size
            user_batch = test_users[start: end]  # 取一部分 user

            all_item = data_generator.all_item

            user_batch = torch.from_numpy(np.array(user_batch)).to(self.device).long()
            all_item = torch.from_numpy(np.array(all_item)).to(self.device).long()

            rate_batch = self.model.rating(user_batch,all_item).detach().cpu()  # shape is [len(user_batch),ITEM_NUM] 即预测评分矩阵

            user_batch_rating_uid = zip(rate_batch.numpy(), user_batch.detach().cpu().numpy())  # 一个user 对应一行评分
            batch_result = pool.map(test_one_user, user_batch_rating_uid)
            count += len(batch_result)

            for re in batch_result:
                result['precision'] += re['precision'] / n_test_users
                result['recall'] += re['recall'] / n_test_users
                result['ndcg'] += re['ndcg'] / n_test_users
                result['hit_ratio'] += re['hit_ratio'] / n_test_users
                result['MAP'] += re['MAP'] / n_test_users
                # result['auc'] += re['auc'] / n_test_users

        assert count == n_test_users
        pool.close()
        return result

    def test_cpp(self, batch_size=256, Ks = [5, 10], mode='val'):

        # data_generator = self.data_generator
        result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
                  'hit_ratio': np.zeros(len(Ks)), 'MAP': np.zeros(len(Ks)), 'auc': 0.}
        met2idx = {'precision': 0, 'recall': 1, 'ndcg': 2, 'hit_ratio': 3, 'MAP': 4}
        u_batch_size = batch_size
        i_batch_size = batch_size

        test_users = list(data_generator.user_test.keys())
        n_test_users = len(test_users)
        n_user_batchs = (n_test_users - 1) // u_batch_size + 1

        count = 0
        # with torch.no_grad():
        all_batch_result = []
        ProxyEvaluator = CPPEvaluator()
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            # end 这里需要完善
            end = (u_batch_id + 1) * u_batch_size
            user_batch = test_users[start: end]  # 取一部分 user
            if len(user_batch)==0:
                break

            all_item = data_generator.all_item

            user_batch = torch.from_numpy(np.array(user_batch)).to(self.device).long()
            all_item = torch.from_numpy(np.array(all_item)).to(self.device).long()

            rate_batch = self.model.rating(user_batch,
                                           all_item).detach().cpu().numpy()  # shape is [len(user_batch),ITEM_NUM] 即预测评分矩阵

            rate_batch[:, 0] = -np.inf
            for idx, user in enumerate(test_users[start: end]):
                train_items = data_generator.user_train[user]
                rate_batch[idx][train_items] = -np.inf
            if mode=='val':
                test_items = [ data_generator.user_valid[u] for u in test_users[start: end]]
            else:
                test_items = [ data_generator.user_test[u] for u in test_users[start: end]]
            batch_result = ProxyEvaluator.eval_score_matrix(rate_batch, test_items, top_k=max(self.Ks),metric=[0,1,2,3,4], thread_num=os.cpu_count()//2)
            count += len(batch_result)
            all_batch_result.append(batch_result)

        all_batch_result = np.concatenate(all_batch_result, axis=0)  # (num_users, metrics_num*max_top)
        final_result = np.mean(all_batch_result, axis=0)  # (1, metrics_num*max_top)
        final_result = np.reshape(final_result, newshape=[5, max(self.Ks)])  # (metrics_num, max_top)
        top_show = [k-1 for k in self.Ks]

        for m,idx in met2idx.items():
            result[m] = final_result[idx][top_show]

        assert count == n_test_users
        return result


class SASTrainer(BaseTrainer):
    def __init__(self,model, lr=0.001, batch_size=500, epochs=15, verbose=5, save_round=200, Ks = [5, 10], early_stop=False,patience=10,
                 device='cpu'):
        super().__init__(model=model, lr=lr, batch_size=batch_size, epochs=epochs, verbose=verbose, save_round=save_round,
                         Ks=Ks, early_stop=early_stop, patience=patience, device=device)
        self.name = 'sastrainer_' + self.model.model_name

    def get_input_data(self):
        users, seqs, pos_items, neg_items = data_generator.seq_sample(self.batch_size)

        users = torch.from_numpy(np.array(users)).to(self.device).long()
        seqs = torch.from_numpy(np.array(seqs)).to(self.device).long()
        pos_items = torch.from_numpy(np.array(pos_items)).to(self.device).long()
        neg_items = torch.from_numpy(np.array(neg_items)).to(self.device).long()

        input_dict = {'users': users, 'seqs':seqs, 'pos_items': pos_items, 'neg_items': neg_items}
        return input_dict

    def get_n_batch(self):
        # 显示 一次epoch需要几个step
        n_users = self.data_generator.n_users
        n_batch = (n_users - 1) // self.batch_size + 1

        # print("Train on {0} samples,  {1} steps per epoch".format(sample_num, n_batch))
        self.logger.info("Train on {0} user,  {1} steps per epoch".format(n_users, n_batch))
        return n_batch


    def test_cpp(self, batch_size=256, Ks = [5, 10], mode='val'):

        # data_generator = self.data_generator
        result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
                  'hit_ratio': np.zeros(len(Ks)), 'MAP': np.zeros(len(Ks)), 'auc': 0.}
        met2idx = {'precision': 0, 'recall': 1, 'ndcg': 2, 'hit_ratio': 3, 'MAP': 4}
        u_batch_size = batch_size
        i_batch_size = batch_size

        test_users = list(data_generator.user_test.keys())
        n_test_users = len(test_users)
        n_user_batchs = (n_test_users - 1) // u_batch_size + 1

        count = 0
        # with torch.no_grad():
        all_batch_result = []
        ProxyEvaluator = CPPEvaluator()
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            # end 这里需要完善
            end = (u_batch_id + 1) * u_batch_size
            user_batch = test_users[start: end]  # 取一部分 user
            if len(user_batch)==0:
                break
            all_item = data_generator.all_item

            user_seq, next_items = self.data_generator.get_test_sample(user_batch, mode=mode)

            all_item = torch.from_numpy(np.array(all_item)).to(self.device).long()
            user_seq = torch.from_numpy(np.array(user_seq)).to(self.device).long()

            rate_batch = self.model.rating(user_seq, all_item).detach().cpu().numpy()  # shape is [len(user_batch),ITEM_NUM] 即预测评分矩阵

            rate_batch[:, 0] = -np.inf
            for idx, user in enumerate(test_users[start: end]):
                train_items = data_generator.user_train[user]
                rate_batch[idx][train_items] = -np.inf
            test_items = next_items

            batch_result = ProxyEvaluator.eval_score_matrix(rate_batch, test_items, top_k=max(self.Ks),metric=[0,1,2,3,4], thread_num=os.cpu_count()//2)
            count += len(batch_result)
            all_batch_result.append(batch_result)

        all_batch_result = np.concatenate(all_batch_result, axis=0)  # (num_users, metrics_num*max_top)
        final_result = np.mean(all_batch_result, axis=0)  # (1, metrics_num*max_top)
        final_result = np.reshape(final_result, newshape=[5, max(self.Ks)])  # (metrics_num, max_top)
        top_show = [k-1 for k in self.Ks]

        for m,idx in met2idx.items():
            result[m] = final_result[idx][top_show]

        assert count == n_test_users
        return result

