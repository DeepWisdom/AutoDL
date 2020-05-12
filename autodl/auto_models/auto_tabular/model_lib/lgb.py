from sklearn.metrics import roc_auc_score
import hyperopt
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict

from autodl.utils.log_utils import log, timeit
from ...auto_tabular.utils.data_utils import ohe2cat
from .meta_model import MetaModel

from ...auto_tabular import CONSTANT

pd.set_option('display.max_rows', 100)


class LGBModel(MetaModel):
    def __init__(self):
        super(LGBModel, self).__init__()
        self.run_num = 0
        self.max_run = 4
        self.rise_num = 0
        self.not_rise_num = 0
        self.not_gain_num = 0

        self.all_data_round_pre = 1
        self.all_data_round = 3
        self.explore_params_round = 2

        self.ensemble_num = 10

        self.not_gain_threhlod = 1

        self.max_length = None
        self.mean = None
        self.std = None

        self.patience = 10

        self.is_init = False

        self.name = 'lgb'
        self.type = 'tree'

        self._model = None

        self.models = {}
        self.ensemble_pred = False

        self.last_preds = None

        self.params = {
            "boosting_type": "gbdt",
            "objective": "multiclass",
            'num_class': 3,
            "metric": "multi_logloss",
            "verbosity": -1,
            "seed": CONSTANT.SEED,
            "num_threads": CONSTANT.JOBS,
        }

        self.hyperparams = {
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'max_bin': 200,
            'subsample': 0.9,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'min_child_weight': 0.001,
            'min_split_gain': 0.02,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            "learning_rate": 0.08,
        }
        self.learning_rates = None

        self.is_multi_label = None

        self.num_class = None

        self.en_models = {}

        self.imp_nums = None

        self.import_cols = None

    def init_model(self, num_class, **kwargs):
        self.is_init = True
        self.params.update({'num_class': num_class})
        self.num_class = num_class

    #@timeit
    def epoch_train(self, dataloader, run_num, is_multi_label=False, info=None, time_remain=None):
        self.is_multi_label = is_multi_label
        X, y, train_idxs, cat = dataloader['X'], dataloader['y'], dataloader['train_idxs'], dataloader['cat_cols']
        train_x, train_y = X.loc[train_idxs], y[train_idxs]

        if info['mode'] == 'bagging':
            self.hyperparams = info['lgb'].copy()
            self.hyperparams['seed'] = np.random.randint(0, 2020)
            num_leaves = self.hyperparams['num_leaves']
            self.hyperparams['num_leaves'] += np.random.randint(-int(num_leaves/10), int(num_leaves/10))
            run_num = 0

        if run_num == self.explore_params_round:
            print('lgb explore_params_round')
            train_x, train_y, val_x, val_y,  = self.split_data(train_x, train_y)

            self.log_feat_importances()

            if train_x.shape[1] > 300 and train_x.shape[0] > 20000:
                train_x = train_x[self.import_cols[:300]]
                val_x = val_x[self.import_cols[:300]]
                log('explore_params_round sample 300 cols')
                train_x.reset_index(drop=True, inplace=True)
                train_x = train_x.sample(n=20000)
                train_y = train_y[list(train_x.index)]
                log('explore_params_round sample 2w samples')

            elif train_x.shape[0] > 20000:
                train_x.reset_index(drop=True, inplace=True)
                train_x = train_x.sample(n=20000)
                train_y = train_y[list(train_x.index)]
                log('explore_params_round sample 2w samples')

            elif train_x.shape[1] > 300:
                train_x = train_x[self.import_cols[:300]]
                val_x = val_x[self.import_cols[:300]]
                log('explore_params_round sample 300 cols')

            print('shape: ', train_x.shape)

            self.bayes_opt(train_x, val_x, train_y, val_y, cat, phase=1)
            self.early_stop_opt(train_x, val_x, train_y, val_y, cat)
            info['lgb'] = self.hyperparams.copy()
            info['imp_cols'] = self.import_cols

        if run_num == self.ensemble_num:
            print('lgb ensemble_num')
            splits = dataloader['splits']
            for i in range(len(splits)):
                train_idxs, val_idxs = splits[i]
                train_x, train_y = X.loc[train_idxs], y[train_idxs]
                hyperparams = self.hyperparams.copy()
                # num_leaves = hyperparams['num_leaves']
                # num_leaves += np.random.randint(-int(num_leaves/10), int(num_leaves/10))
                # hyperparams['num_leaves'] = num_leaves
                # log('model {} leaves {}'.format(i, num_leaves))
                if self.is_multi_label:
                    self.en_models = defaultdict(list)
                    for cls in range(self.num_class):
                        cls_y = train_y[:, cls]
                        lgb_train = lgb.Dataset(train_x, cls_y)
                        if not self.learning_rates:
                            self.en_models[i].append(lgb.train({**self.params, **hyperparams}, train_set=lgb_train))
                        else:
                            self.en_models[i].append(
                                lgb.train({**self.params, **hyperparams},
                                          train_set=lgb_train, learning_rates=self.learning_rates))
                else:
                    lgb_train = lgb.Dataset(train_x, ohe2cat(train_y))
                    if not self.learning_rates:
                        self.en_models[i] = lgb.train({**self.params, **hyperparams}, train_set=lgb_train)
                    else:
                        self.en_models[i] = lgb.train({**self.params, **hyperparams},
                                                   train_set=lgb_train, learning_rates=self.learning_rates)
                self.ensemble_pred = True

        else:
            print('lgb norm train')
            train_x, train_y = X.loc[train_idxs], y[train_idxs]
            hyperparams = self.hyperparams.copy()
            log('hyperparams {}'.format(hyperparams))
            if run_num == self.all_data_round_pre or run_num == self.all_data_round:
                print('lgb all data round')
                all_train_idxs = dataloader['all_train_idxs']
                train_x = X.loc[all_train_idxs]
                train_y = y[all_train_idxs]
            print('shape: ', train_x.shape)
            if not is_multi_label:
                lgb_train = lgb.Dataset(train_x, ohe2cat(train_y))
                if not self.learning_rates:
                    self._model = lgb.train({**self.params, **hyperparams}, train_set=lgb_train)
                else:
                    self._model = lgb.train({**self.params, **hyperparams},
                                            train_set=lgb_train, learning_rates=self.learning_rates)
            else:
                self.params['num_class'] = 2
                for cls in range(self.num_class):
                    cls_y = train_y[:, cls]
                    lgb_train = lgb.Dataset(train_x, cls_y)
                    if not self.learning_rates:
                        self.models[cls] = lgb.train({**self.params, **self.hyperparams}, train_set=lgb_train)
                    else:
                        self.models[cls] = lgb.train({**self.params, **self.hyperparams},
                                                     train_set=lgb_train, learning_rates=self.learning_rates)
            self.log_feat_importances()
            if self.imp_nums is not None:
                info['imp_nums'] = self.imp_nums


    #@timeit
    def epoch_valid(self, dataloader):
        X, y, val_idxs= dataloader['X'], dataloader['y'], dataloader['val_idxs']
        val_x, val_y = X.loc[val_idxs], y[val_idxs]
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
        if not self.ensemble_pred:
            if not self.is_multi_label:
                preds = self._model.predict(test_x)
            else:
                all_preds = []
                for cls in range(self.num_class):
                    preds = self.models[cls].predict(test_x)
                    all_preds.append(preds[:,1])
                preds = np.stack(all_preds, axis=1)

            self.last_preds = preds
            return preds
        else:
            ensemble_preds = 0
            preds = 0
            for i, model in self.en_models.items():
                if not self.is_multi_label:
                    preds = model.predict(test_x)
                else:
                    all_preds = []
                    for cls in range(self.num_class):
                        preds = model[cls].predict(test_x)
                        all_preds.append(preds[:,1])
                    preds = np.stack(all_preds, axis=1)
                m = np.mean(preds)
                preds = preds/m/5
                ensemble_preds += preds
            last_mean = np.mean(self.last_preds)
            last_preds = self.last_preds/last_mean
            return 0.7*last_preds + 0.3*preds

    def log_feat_importances(self, return_info=False):
        if not self.is_multi_label:
            importances = pd.DataFrame({'features': [i for i in self._model.feature_name()],
                                        'importances': self._model.feature_importance("gain")})
        else:
            importances = pd.DataFrame({'features': [i for i in self.models[0].feature_name()],
                                        'importances': self.models[0].feature_importance("gain")})

        importances.sort_values('importances', ascending=False, inplace=True)


        importances = importances[importances['importances'] > 0]

        size = int(len(importances)*0.8)


        self.import_cols = importances['features'][:size].values

        self.imp_nums = list(importances['features'][:30].values)

    #@timeit
    def bayes_opt(self, X_train, X_eval, y_train, y_eval, categories, phase=1):
        if self.is_multi_label:
            train_data = lgb.Dataset(X_train, label=y_train[:,0])
            valid_data = lgb.Dataset(X_eval, label=y_eval[:,0])
        else:
            y_train = ohe2cat(y_train)
            y_eval = ohe2cat(y_eval)
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_eval, label=y_eval)

        params = self.params

        if phase == 1:
            space = {
                'max_depth': hp.choice("max_depth", [-1, 5, 7, 9]),
                "num_leaves": hp.choice("num_leaves", np.linspace(20, 61, 10, dtype=int)),
                "reg_alpha": hp.uniform("reg_alpha", 0, 1),
                "reg_lambda": hp.uniform("reg_lambda", 0, 1),
                "min_child_samples": hp.choice("min_data_in_leaf", np.linspace(10, 120, 10, dtype=int)),
                "min_child_weight": hp.uniform('min_child_weight', 0.01, 1),
                "min_split_gain": hp.uniform('min_split_gain', 0.001, 0.1),
                'colsample_bytree': hp.choice("colsample_bytree", [0.7, 0.9]),
                "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.1)),
            }
            tmp_hyperparams = {}
            tmp_hyperparams['num_boost_round'] = 100
            max_evals = 20

        else:
            space = {
                "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.1)),
            }
            tmp_hyperparams = {}
            update = ['max_depth', 'num_leaves', 'reg_alpha', 'reg_lambda', 'min_data_in_leaf', 'min_child_weight', 'min_split_gain']

            for p in update:
                tmp_hyperparams[p] = self.hyperparams[p]

            tmp_hyperparams['num_boost_round'] = 500
            max_evals = 5

        def objective(hyperparams):
            tmp_hyperparams.update(hyperparams)
            model = lgb.train({**params, **tmp_hyperparams}, train_set=train_data, valid_sets=valid_data,
                              #categorical_feature=categories,
                              early_stopping_rounds=18, verbose_eval=0)

            score = model.best_score["valid_0"][params["metric"]]

            # in classification, less is better
            return {'loss': score, 'status': STATUS_OK}

        trials = Trials()
        best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                             algo=tpe.suggest, max_evals=max_evals, verbose=1,
                             rstate=np.random.RandomState(1))
        self.hyperparams.update(space_eval(space, best))

    #@timeit
    def early_stop_opt(self, X_train, X_eval, y_train, y_eval, categories):
        if self.is_multi_label:
            lgb_train = lgb.Dataset(X_train, y_train[:, 0])
            lgb_eval = lgb.Dataset(X_eval, y_eval[:, 0], reference=lgb_train)
        else:
            y_train = ohe2cat(y_train)
            y_eval = ohe2cat(y_eval)
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)

        self.hyperparams['num_boost_round'] = 1000
        tmp_lr = self.hyperparams.pop('learning_rate')
        self.learning_rates = self.get_log_lr(1000, tmp_lr, tmp_lr*0.6)

        self._model = lgb.train({**self.params, **self.hyperparams}, verbose_eval=20,
                          train_set=lgb_train, valid_sets=lgb_eval, valid_names='eval',
                          early_stopping_rounds=20, learning_rates=self.learning_rates) #categorical_feature=categories)

        self.hyperparams['num_boost_round'] = self._model.best_iteration
        self.learning_rates = self.learning_rates[:self._model.best_iteration]


    def get_log_lr(self,num_boost_round,max_lr,min_lr):
        learning_rates = [max_lr+(min_lr-max_lr)/np.log(num_boost_round)*np.log(i) for i in range(1,num_boost_round+1)]
        return learning_rates


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
