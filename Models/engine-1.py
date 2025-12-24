import sys
import numpy as np, pandas as pd
import math
import random
import time
import tqdm
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn import metrics


class Engine(object):
    def __init__(self, TrainSettings):
        self.TrainSettings = TrainSettings
        # self._writer = SummaryWriter(logdir=ResultSettings['save_dir'], comment=ModelSettings['model_name'])
        self.optimizer = self._get_optimizer()
        self.model.Loss = self._get_criterion()
        self.logloss = torch.nn.BCEWithLogitsLoss()
        self.device = TrainSettings['device']
        self.clip_grad_norm = False
        if 'max_grad_norm' in TrainSettings:
            self.clip_grad_norm = True
            self.max_grad_norm = eval(TrainSettings['max_grad_norm'])

    def _get_optimizer(self):
        optimizer_type = self.TrainSettings['optimizer']
        learning_rate = eval(self.TrainSettings['learning_rate'])
        weight_decay = eval(self.TrainSettings['weight_decay'])

        params = self.model.parameters()
        if optimizer_type == 'adam':
            return torch.optim.Adam(params=params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'adagrad':
            return torch.optim.Adagrad(params=params, lr=learning_rate, lr_decay=0, weight_decay=weight_decay,
                                       initial_accumulator_value=0)
        elif optimizer_type == 'rmsprop':
            return torch.optim.RMSprop(params=params, lr=learning_rate, alpha=0.9)
        else:
            raise ValueError('unknow optimizer_type name: ' + optimizer_type)

    def _get_criterion(self):
        criterion_type = self.TrainSettings['criterion']
        if criterion_type == 'mse':
            return torch.nn.MSELoss()
        elif criterion_type == 'bce':
            return torch.nn.BCELoss()
        elif criterion_type == 'bcelogits':
            return torch.nn.BCEWithLogitsLoss()
        elif criterion_type == 'ce':
            return torch.nn.CrossEntropyLoss()
        else:
            raise ValueError('unknow criterion_type name: ' + criterion_type)

    def _get_auc(self, pred_score, label):
        return metrics.roc_auc_score(label, pred_score), None, None

    def _get_f1(self, pred_score, label, threshold=0.5):
        pred_label = (np.array(pred_score) >= threshold).astype(int)
        return metrics.f1_score(label, pred_label)

    def _get_recall(self, pred_score, label, threshold=0.5):
        pred_label = (np.array(pred_score) >= threshold).astype(int)
        return metrics.recall_score(label, pred_label)

    def _get_mse(self, pred_score, label):
        return metrics.mean_squared_error(label, pred_score)


class CTREngine(Engine):
    def __init__(self, model, TrainSettings):
        self.model = model
        self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1, 2, 7])
        self.model.to(TrainSettings['device'])
        super(CTREngine, self).__init__(TrainSettings)

    def train(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        device = self.device

        loss_sum = 0
        loss_array = []
        t0 = time.time()

        for i, input_list in enumerate(tqdm.tqdm(train_loader, desc="train", smoothing=0, mininterval=1.0)):

            # run model
            input_list = input_list[:-1]
            input_list = [x.long().to(device) for x in input_list]
            labels = input_list[0]
            self.optimizer.zero_grad()
            logits = self.model(input_list[1])
            # logits = self.model(input_list[1],epoch_id)
            loss_list = self.model.loss(logits, labels)

            # loss
            with torch.autograd.set_detect_anomaly(True):
                loss = sum(loss_list)
                loss.backward(retain_graph=False)
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm, norm_type=2)
                self.optimizer.step()
            loss_array.append(loss.item())
            loss_sum += loss.item()

        t1 = time.time()
        print("Epoch ", epoch_id, " Train cost:", t1 - t0, " Loss: ", np.mean(loss_array))
        return np.mean(loss_array)

    def evaluate(self, test_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        device = self.device

        # 初始化4个空列表，分别存储测试集的logits，预测分数、真实标签和页面ID
        test_logits, test_scores, test_labels, test_page_ids = [], [], [], []

        for i, input_list in enumerate(tqdm.tqdm(test_loader, desc="eval test", smoothing=0, mininterval=1.0)):
            batch_label = input_list[0]
            batch_page_id = input_list[-1]  # search_id 作为 page_id
            batch_input = input_list[1]

            test_labels.extend(batch_label)
            test_page_ids.extend(batch_page_id)

            batch_input = batch_input.long().to(device)
            pred_logits = self.model(batch_input)
            test_logits.extend(list(pred_logits.data.cpu().numpy()))
            pred_score = torch.sigmoid(pred_logits)  # 用sigmoid得到预测分数
            test_scores.extend(list(pred_score.data.cpu().numpy()))

        # 转换为 numpy 数组
        test_scores_np = np.array(test_scores)
        test_labels_np = np.array(test_labels)
        test_page_ids_np = np.array(test_page_ids)

        # 计算整体指标
        auc = self._get_auc(test_scores_np, test_labels_np)[0]
        f1 = self._get_f1(test_scores_np, test_labels_np)
        recall = self._get_recall(test_scores_np, test_labels_np)
        mse = self._get_mse(test_scores_np, test_labels_np)

        # 按 page_id 分组计算每个页面的 AUC
        # 按 page_id 分组计算每个页面的 AUC
        page_aucs = []
        unique_page_ids = np.unique(test_page_ids_np)

        for page_id in unique_page_ids:
            idxs = test_page_ids_np == page_id
            page_labels = test_labels_np[idxs]

            # 只有当页面中同时存在正负样本时才计算 AUC
            if len(np.unique(page_labels)) < 2:
                continue

            page_auc = metrics.roc_auc_score(page_labels, test_scores_np[idxs])
            page_aucs.append(page_auc)

        pv_auc = np.mean(page_aucs) if page_aucs else 0.0  # 防止空列表导致报错

        t1 = time.time()

        test_result = ''
        test_result += "AUC: " + str(auc) + '\n'
        test_result += "F1: " + str(f1) + '\n'
        test_result += "Recall: " + str(recall) + '\n'
        test_result += "MSE: " + str(mse) + '\n'
        test_result += "PV_AUC: " + str(pv_auc) + '\n'

        print(
            "evaluate test cost: ", t1 - time.time(),
                                    " AUC: " + str(auc),
                                    " F1: " + str(f1),
                                    " Recall: " + str(recall),
                                    " MSE: " + str(mse),
                                    " PV_AUC: " + str(pv_auc)
        )

        return test_result, auc, f1, recall, mse, pv_auc



