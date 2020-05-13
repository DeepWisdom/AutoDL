import xgboost as xgb
from sklearn.metrics import roc_auc_score
import hyperopt
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

from .meta_model import MetaModel
from ...auto_tabular import CONSTANT
from ...auto_tabular.utils.data_utils import ohe2cat


class XGBModel(MetaModel):

    def __init__(self):
        super(XGBModel, self).__init__()
        self.max_run = 2
        self.all_data_round = 1
        self.explore_params_round = 0

        self.not_gain_threhlod = 3

        self.patience = 3

        self.is_init = False

        self.name = 'xgb'
        self.type = 'tree'

        self._model = None

        self.params = {
            "boosting_type": "gbdt",
            "objective": "multi:softprob",
            "nthread": CONSTANT.JOBS,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
            "seed": CONSTANT.SEED,
        }

        self.hyperparams = {
            "learning_rate": 0.02,
            "max_depth": 6,
            "min_child_weight": 0.01,
            "min_data_in_leaf": 100,
            "gamma": 0.1,
            "lambda": 0.1,
            "alpha": 0.1}

        self.is_multi_label = None

        self.num_class = None

        self.models = {}

        self.import_cols = None

    def init_model(self, num_class, **kwargs):
        self.is_init = True
        self.params.update({'num_class': num_class})
        self.num_class = num_class

    #@timeit
    def epoch_train(self, dataloader, run_num, is_multi_label=None, info=None, time_remain=None):
        self.is_multi_label = is_multi_label
        X, y, train_idxs, cat = dataloader['X'], dataloader['y'], dataloader['train_idxs'], dataloader['cat_cols']
        train_x, train_y = X.loc[train_idxs], y[train_idxs]

        if info['mode'] == 'bagging':
            self.hyperparams = info['xgb'].copy()
            self.hyperparams['random_seed'] = np.random.randint(0, 2020)
            run_num = self.explore_params_round

        if run_num == self.explore_params_round:
            print('xgb explore_params_round')
            train_x, train_y, val_x, val_y, = self.split_data(train_x, train_y)

            self.import_cols = info['imp_cols']

            if train_x.shape[1] > 300 and train_x.shape[0] > 10000:
                train_x = train_x[self.import_cols[:300]]
                val_x = val_x[self.import_cols[:300]]
                train_x.reset_index(drop=True, inplace=True)
                train_x = train_x.sample(n=10000)
                train_y = train_y[list(train_x.index)]

            elif train_x.shape[0] > 10000:
                train_x.reset_index(drop=True, inplace=True)
                train_x = train_x.sample(n=10000)
                train_y = train_y[list(train_x.index)]

            elif train_x.shape[1] > 300:
                train_x = train_x[self.import_cols[:300]]
                val_x = val_x[self.import_cols[:300]]

            self.bayes_opt(train_x, val_x, train_y, val_y, cat)
            self.early_stop_opt(train_x, val_x, train_y, val_y, cat)
            info['xgb'] = self.hyperparams.copy()

        train_x, train_y = X.loc[train_idxs], y[train_idxs]
        if run_num == self.all_data_round:
            all_train_idxs = dataloader['all_train_idxs']
            train_x = X.loc[all_train_idxs]
            train_y = y[all_train_idxs]
        if not self.is_multi_label:
            xgb_train = xgb.DMatrix(train_x, ohe2cat(train_y))
            self._model = xgb.train({**self.params, **self.hyperparams}, xgb_train)
        else:
            for cls in range(self.num_class):
                cls_y = train_y[:, cls]
                xgb_train = xgb.DMatrix(train_x, cls_y)
                self.models[cls] = self._model = xgb.train({**self.params, **self.hyperparams}, xgb_train)


    #@timeit
    def epoch_valid(self, dataloader):
        X, y, val_idxs= dataloader['X'], dataloader['y'], dataloader['val_idxs']
        val_x, val_y = X.loc[val_idxs], y[val_idxs]
        val_x = xgb.DMatrix(val_x)
        if not self.is_multi_label:
            preds = self._model.predict(val_x)
        else:
            all_preds = []
            for cls in range(y.shape[1]):
                preds = self.models[cls].predict(val_x)
                all_preds.append(preds[:,1])
            preds = np.stack(all_preds, axis=1)
        valid_auc = roc_auc_score(val_y, preds)
        return valid_auc

    #@timeit
    def predict(self, dataloader):
        X, test_idxs = dataloader['X'], dataloader['test_idxs']
        test_x = X.loc[test_idxs]
        test_x = xgb.DMatrix(test_x)
        if not self.is_multi_label:
            return self._model.predict(test_x)
        else:
            all_preds = []
            for cls in range(self.num_class):
                preds = self.models[cls].predict(test_x)
                all_preds.append(preds[:, 1])
            return np.stack(all_preds, axis=1)

    #@timeit
    def bayes_opt(self, X_train, X_eval, y_train, y_eval, categories):
        if self.is_multi_label:
            dtrain = xgb.DMatrix(X_train, y_train[:, 1])
            dvalid = xgb.DMatrix(X_eval, y_eval[:, 1])
        else:
            dtrain = xgb.DMatrix(X_train, ohe2cat(y_train))
            dvalid = xgb.DMatrix(X_eval, ohe2cat(y_eval))
        space = {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.1)),
            "max_depth": hp.choice("max_depth", [4, 6, 8, 10, 12]),
            "min_child_weight": hp.uniform('min_child_weight', 0.01, 1),
            "min_data_in_leaf": hp.choice("min_data_in_leaf", np.linspace(10, 100, 20, dtype=int)),
            "gamma": hp.uniform("gamma", 0.001, 0.1),
            "lambda": hp.uniform("lambda", 0, 1),
            "alpha": hp.uniform("alpha", 0, 1),
            "colsample_bytree": hp.choice("colsample_bytree", [0.7, 0.9]),
            "colsample_bylevel": hp.choice("colsample_bylevel", [0.7, 0.9]),
            "colsample_bynode": hp.choice("colsample_bynode", [0.7, 0.9]),

        }

        def objective(hyperparams):
            model = xgb.train({**self.params, **hyperparams}, dtrain,  num_boost_round=50)

            pred = model.predict(dvalid)
            if self.is_multi_label:
                score = roc_auc_score(y_eval[:, 1], pred[:, 1])
            else:
                score = roc_auc_score(y_eval, pred)

            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()
        best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                             algo=tpe.suggest, max_evals=10, verbose=1,
                             rstate=np.random.RandomState(1))

        self.hyperparams.update(space_eval(space, best))

    def early_stop_opt(self, X_train, X_eval, y_train, y_eval, categories):
        if self.is_multi_label:
            dtrain = xgb.DMatrix(X_train, y_train[:, 1])
            dvalid = xgb.DMatrix(X_eval, y_eval[:, 1])
        else:
            dtrain = xgb.DMatrix(X_train, ohe2cat(y_train))
            dvalid = xgb.DMatrix(X_eval, ohe2cat(y_eval))

        model = xgb.train({**self.params, **self.hyperparams}, dtrain, evals=[(dvalid, 'eval')], num_boost_round=1200,
                          early_stopping_rounds=10) #categorical_feature=categories)


        self.params['num_boost_round'] = model.best_iteration

    def split_data(self, x, y):
        new_x = x.copy()
        new_x.reset_index(drop=True, inplace=True)
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
        self.splits = {}
        i = 0
        for train_idxs, val_idxs in sss.split(new_x, y):
            self.splits[i] = [train_idxs, val_idxs]
            i += 1
        new_train_x = new_x.loc[self.splits[0][0]]
        new_train_y = y[self.splits[0][0]]

        new_val_x = new_x.loc[self.splits[0][1]]
        new_val_y = y[self.splits[0][1]]

        return new_train_x, new_train_y, new_val_x, new_val_y


