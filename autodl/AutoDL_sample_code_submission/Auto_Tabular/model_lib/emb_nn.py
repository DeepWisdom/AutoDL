import librosa
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Activation, Add

from Auto_Tabular.CONSTANT import *
from Auto_Tabular.utils.data_utils import ohe2cat
from .meta_model import MetaModel
from sklearn.metrics import roc_auc_score

from fastai.tabular import *

import pytorch as torch

import fastai
from fastai.text import *


class ENNModel(MetaModel):
    def __init__(self):
        super(ENNModel, self).__init__()
        #clear_session()
        self.max_length = None
        self.mean = None
        self.std = None

        self._model = None
        self.is_init = False

        self.name = 'enn'
        self.type = 'emb_nn'
        self.patience = 50

        self.max_run = 100

        self.all_data_round = 80

        self.not_gain_threhlod = 50

        self.data_gen = None

    def init_model(self,
                   num_classes,
                   shape=None,
                   **kwargs):

        self.is_init = True

    def epoch_train(self, dataloader, run_num):
        if self.data_gen == None:
            X, y, cats = dataloader['X'], dataloader['y'], dataloader['cat_cols']
            all_train_idxs, val_idxs, test_idxs = dataloader['all_train_idxs'],  dataloader['val_idxs'], dataloader['test_idxs']
            all_x, all_y, test_x = X.loc[all_train_idxs], y[all_train_idxs], X.loc[test_idxs]
            all_x['target'] = all_y
            procs = []
            path = './'
            num_cols = [col for col in X.columns if col not in cats]

            self.data_gen = TabularDataBunch.from_df(
                path, all_x, 'target', valid_idx=val_idxs, procs=procs, cat_names=cats, cont_names=num_cols)
            emb_szs = {col: 5 for col in cats}
            self._model = tabular_learner(self.data_gen , layers=[200, 100], emb_szs=emb_szs, metrics=accuracy)

        self._model.fit(epochs=5)


    def epoch_valid(self, dataloader):
        X, y, val_idxs= dataloader['X'], dataloader['y'], dataloader['val_idxs']
        val_x, val_y = X.loc[val_idxs].values, y[val_idxs]
        preds = self._model.predict(val_x)
        preds = preds.cpu().numpy()[:, 1]
        valid_auc = roc_auc_score(val_y, preds)
        return valid_auc

    def predict(self, dataloader, batch_size=32):
        X, test_idxs = dataloader['X'], dataloader['test_idxs']
        test_x = X.loc[test_idxs].values
        preds = self._model.predict(test_x)
        preds = preds.cpu().numpy()[:, 1]
        return preds



def auroc_score(input, target):
    input, target = input.cpu().numpy()[:, 1], target.cpu().numpy()
    return roc_auc_score(target, input)


# Callback to calculate AUC at the end of each epoch
# class AUROC(Callback):
#     _order = -20  # Needs to run before the recorder
#
#     def __init__(self, learn, **kwargs):
#         self.learn = learn
#
#     def on_train_begin(self, **kwargs):
#         self.learn.recorder.add_metric_names(['AUROC'])
#
#     def on_epoch_begin(self, **kwargs):
#         self.output, self.target = [], []
#
#     def on_batch_end(self, last_target, last_output, train, **kwargs):
#         if not train:
#             self.output.append(last_output)
#             self.target.append(last_target)
#
#     def on_epoch_end(self, last_metrics, **kwargs):
#         if len(self.output) > 0:
#             output = torch.cat(self.output)
#             target = torch.cat(self.target)
#             preds = F.softmax(output, dim=1)
#             metric = auroc_score(preds, target)
#             return add_metrics(last_metrics, [metric])


