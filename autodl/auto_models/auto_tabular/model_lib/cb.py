from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import hyperopt
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from ...auto_tabular import CONSTANT
from ...auto_tabular.utils.data_utils import ohe2cat
from .meta_model import MetaModel


class CBModel(MetaModel):

    def __init__(self):
        super(CBModel, self).__init__()
        self.max_run = 2
        self.all_data_round = 1
        self.explore_params_round = 0

        self.not_gain_threhlod = 3

        self.patience = 3

        self.is_init = False

        self.name = 'cb'
        self.type = 'tree'

        self._model = None

        self.params = {
            'task_type': 'GPU',
            "loss_function": "MultiClass",
            "random_seed": CONSTANT.SEED,
            'verbose': False
        }

        self.hyperparams = {
            "learning_rate": 0.02,
            'iterations': 1200,
        }

        self.is_multi_label = None

        self.num_class = None

        self.models = {}

    def init_model(self, num_class, **kwargs):
        self.is_init = True
        self.num_class = num_class

    #@timeit
    def epoch_train(self, dataloader, run_num, is_multi_label=None, info=None, time_remain=None):
        self.is_multi_label = is_multi_label
        X, y, train_idxs, cat = dataloader['X'], dataloader['y'], dataloader['train_idxs'], dataloader['cat_cols']
        train_x, train_y = X.loc[train_idxs], y[train_idxs]

        if info['mode'] == 'bagging':
            self.hyperparams = info['cb'].copy()
            self.hyperparams['random_seed'] = np.random.randint(0, 2020)
            run_num = self.explore_params_round


        if run_num == self.explore_params_round:
            train_x, train_y, val_x, val_y, = self.split_data(train_x, train_y)

            self.import_cols = info['imp_cols']

            if train_x.shape[1] > 300 and train_x.shape[0] > 20000:
                train_x = train_x[self.import_cols[:300]]
                val_x = val_x[self.import_cols[:300]]
                train_x.reset_index(drop=True, inplace=True)
                train_x = train_x.sample(n=20000)
                train_y = train_y[list(train_x.index)]

            elif train_x.shape[0] > 20000:
                train_x.reset_index(drop=True, inplace=True)
                train_x = train_x.sample(n=20000)
                train_y = train_y[list(train_x.index)]

            elif train_x.shape[1] > 300:
                train_x = train_x[self.import_cols[:300]]
                val_x = val_x[self.import_cols[:300]]

            self.bayes_opt(train_x, val_x, train_y, val_y, cat)
            self.early_stop_opt(train_x, val_x, train_y, val_y, cat)

            info['cb'] = self.hyperparams.copy()

        train_x, train_y = X.loc[train_idxs], y[train_idxs]
        if run_num == self.all_data_round:
            all_train_idxs = dataloader['all_train_idxs']
            train_x = X.loc[all_train_idxs]
            train_y = y[all_train_idxs]

        if self.is_multi_label:
            for cls in range(self.num_class):
                cls_y = train_y[:, cls]
                self.models[cls] = CatBoostClassifier(**{**self.params, **self.hyperparams})
                self.models[cls].fit(train_x, cls_y)
        else:
            self._model = CatBoostClassifier(**{**self.params, **self.hyperparams})
            self._model.fit(train_x, ohe2cat(train_y))

    #@timeit
    def epoch_valid(self, dataloader):
        X, y, val_idxs= dataloader['X'], dataloader['y'], dataloader['val_idxs']
        val_x, val_y = X.loc[val_idxs], y[val_idxs]
        if not self.is_multi_label:
            preds = self._model.predict_proba(val_x)
        else:
            all_preds = []
            for cls in range(y.shape[1]):
                preds = self.models[cls].predict_proba(val_x)
                all_preds.append(preds[:,1])
            preds = np.stack(all_preds, axis=1)
        valid_auc = roc_auc_score(val_y, preds)
        return valid_auc

    #@timeit
    def predict(self, dataloader):
        X, test_idxs = dataloader['X'], dataloader['test_idxs']
        test_x = X.loc[test_idxs]
        if not self.is_multi_label:
            return self._model.predict_proba(test_x)
        else:
            all_preds = []
            for cls in range(self.num_class):
                preds = self.models[cls].predict_proba(test_x)
                all_preds.append(preds[:, 1])
            return np.stack(all_preds, axis=1)

    def bayes_opt(self, X_train, X_eval, y_train, y_eval, categories):
        if self.is_multi_label:
            y_train = y_train[:, 1]
            y_eval = y_eval[:, 1]
        else:
            y_train = ohe2cat(y_train)

        space = {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.1)),
            "depth": hp.choice("depth", [4, 6, 8, 10, 12]),
            "l2_leaf_reg": hp.uniform('l2_leaf_reg', 0.1, 2),

        }

        def objective(hyperparams):
            hyperparams = self.hyperparams.copy()
            hyperparams['iterations'] = 300
            model = CatBoostClassifier(**{**self.params, **hyperparams})
            model.fit(X_train, y_train)
            pred = model.predict_proba(X_eval)

            if self.is_multi_label:
                score = roc_auc_score(y_eval, pred[:, 1])
            else:
                score = roc_auc_score(y_eval, pred)

            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()
        best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                             algo=tpe.suggest, max_evals=15, verbose=1,
                             rstate=np.random.RandomState(1))

        self.hyperparams.update(space_eval(space, best))

    def early_stop_opt(self, X_train, X_eval, y_train, y_eval, categories):
        if self.is_multi_label:
            y_train = y_train[:, 1]
            y_eval = y_eval[:, 1]
        else:
            y_train = ohe2cat(y_train)
            y_eval = ohe2cat(y_eval)

        model = CatBoostClassifier(**{**self.params, **self.hyperparams})
        model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)],
                  use_best_model=True, verbose=10, early_stopping_rounds=20)

        self.params['iterations'] = model.best_iteration_

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