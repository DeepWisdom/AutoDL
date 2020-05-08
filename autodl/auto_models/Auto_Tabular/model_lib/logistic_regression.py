import numpy as np
from sklearn.linear_model import logistic
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score

from CONSTANT import *
from .meta_model import MetaModel
from Auto_Tabular.utils.log_utils import log
from Auto_Tabular.utils.data_utils import ohe2cat


class LogisticRegression(MetaModel):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self._model = None
        self.is_init = False

        self.name = 'lr'
        self.type = 'lr'
        self.patience = 1

        self.max_run = 1

        self.all_data_round = 1

    def init_model(self, class_num):
        self._model = logistic.LogisticRegression(
            C=1.0, max_iter=200, solver='liblinear', multi_class='auto')
        self.is_init = True

    def epoch_train(self, dataloader, run_num):
        X, y, train_idxs = dataloader['X'], dataloader['y'], dataloader['train_idxs']
        train_x, train_y = X.loc[train_idxs], y[train_idxs]
        self._model.fit(train_x, ohe2cat(train_y))

    def epoch_valid(self, dataloader):
        X, y, val_idxs= dataloader['X'], dataloader['y'], dataloader['val_idxs']
        val_x, val_y = X.loc[val_idxs], y[val_idxs]
        preds = self._model.predict_proba(val_x)
        valid_auc = roc_auc_score(val_y, preds)
        return valid_auc

    def predict(self, dataloader, batch_size=32):
        X, test_idxs = dataloader['X'], dataloader['test_idxs']
        test_x = X.loc[test_idxs]
        return self._model.predict_proba(test_x)




