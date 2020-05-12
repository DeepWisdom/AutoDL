import numpy as np
import gc
import collections
from tensorflow.python.keras import backend as K

from ..auto_tabular.feature.feat_engine import FeatEngine


class Explore:
    def __init__(self, metadata, info, model_space, data_space):
        self.metadata = metadata
        self.info = info
        self.info['mode'] = 'first_round'
        self.model_space = model_space
        self.data_space = data_space

        self.model = None

        self.model_prior = model_space.model_prior
        self.model_idx = 0

        self.input_shape = None

        self.patience = 3
        self.auc_gain_threshold = 1e-4
        self.ensemble_std_threshold = 1e-2

        self.round_num = 1

        self.hist_info = {}

        self.dataloader = None

        self.feat_engine = FeatEngine()

        self.update_predcit = True

        self.use_all_data = False

    def explore_space(self, train_loop_num, time_remain=None):
        self.explore_model_space(train_loop_num)
        self.explore_data_space(train_loop_num)
        self.create_model(self.metadata.output_dim)

        # train and evaluate
        self.model.epoch_train(self.dataloader, run_num=self.model.run_num,
                               is_multi_label=self.info['is_multi_label'], info=self.info, time_remain=time_remain)


        if not self.use_all_data:
            val_auc = self.model.epoch_valid(self.dataloader)

        else:
            val_auc = self.model.best_auc+0.0001
            self.use_all_data = False

        self.update_model_hist(val_auc)

    def explore_model_space(self, train_loop_num):
        if train_loop_num == 1:
            self.model = self.model_space.get_model(self.model_prior[self.model_idx], self.round_num)
            self.last_model_type = self.model.type
        else:
            if self.model.not_rise_num == self.model.patience \
                    or (self.model.not_gain_num > self.model.not_gain_threhlod) \
                    or self.model.run_num >= self.model.max_run or self.info['mode'] =='bagging':
                self.model_idx += 1
                self.reset_model_cache()
                if self.model_idx == len(self.model_prior):
                    self.sort_model_prior()
                    self.info['mode'] = 'bagging'
                    self.data_space.update = True
                self.model = self.model_space.get_model(self.model_prior[self.model_idx], self.round_num)
                self.use_all_data = False
                if self.model.type != self.last_model_type:
                    self.dataloader = None
                    gc.collect()

    def explore_data_space(self, train_loop_num):
        self.feat_engine.fit_transform(self.data_space, train_loop_num, info=self.info)

        if self.data_space.update or self.dataloader is None:
            self.dataloader = self.data_space.get_dataloader(train_loop_num=train_loop_num,
                                                             round_num=self.round_num,
                                                             run_num=self.model.run_num,
                                                             use_all_data=self.use_all_data,
                                                             model_type=self.model.type)
            self.data_space.update = False

    def update_model_hist(self, val_auc):
        self.model.run_num += 1
        self.model.auc_gain = val_auc - self.model.hist_auc[-1]
        if self.model.auc_gain < self.auc_gain_threshold:
            self.model.not_gain_num += 1
        else:
            self.model.not_gain_num = 0
        self.model.hist_auc.append(val_auc)
        if val_auc > self.model.best_auc:
            self.model.best_auc = val_auc
            self.update_predcit = True
        else:
            self.update_predcit = False
            self.model.not_rise_num += 1

        if self.model.run_num >= self.model.all_data_round or self.model.not_gain_num > 3:
            self.use_all_data = True
        else:
            self.use_all_data = False

        if hasattr(self.model, 'all_data_round_pre'):
            if self.model.run_num == self.model.all_data_round_pre:
                self.use_all_data = True

    def reset_model_cache(self):
        del self.model
        self.model = None
        gc.collect()
        K.clear_session()

    def create_model(self, class_num):
        if not self.model.is_init:
            if self.model.type == 'nn_keras':
                self.model.init_model(class_num, shape=self.dataloader['shape'], is_multi_label=self.info['is_multi_label'])
            else:
                self.model.init_model(class_num)

    def sort_model_prior(self):
        model_perform = collections.defaultdict(list)
        for name, info in self.hist_info.items():
            first_name = name.split('_')[0]
            auc = info[0]
            if first_name in model_perform:
                model_perform[first_name].append(auc)
        self.model_prior = sorted(self.model_prior, key=lambda x: np.mean(model_perform[x]), reverse=True)
        self.model_idx = 0
        self.round_num += 1

    def get_top_preds(self):
        models_name = self.hist_info.keys()
        models_auc = [self.hist_info[name][0] for name in models_name]
        models_name_sorted, models_auc_sored = (list(i) for i in
                                                zip(*sorted(zip(models_name, models_auc), key=lambda x: x[1], reverse=True)))

        for i in range(len(models_auc_sored), 0, -1):
            std = np.std(models_auc_sored[:i])
            top_num = i
            if std < self.ensemble_std_threshold:
                break

        top_auc = np.array(models_auc_sored[:top_num])
        # weights = top_auc / top_auc.sum()
        # print(weights)

        top_auc = top_auc + 15*(top_auc - top_auc.mean())
        top_auc = np.array([max(0.01, i) for i in top_auc])
        weights = top_auc / top_auc.sum()
        print(weights)

        top_preds = []
        for i in range(top_num):
            name = models_name_sorted[i]
            rank = i + 1
            auc = models_auc_sored[i]
            weight = weights[i]
            preds = self.hist_info[name][1]
            top_preds.append((name, rank, auc, weight, preds))
        return top_preds

    def predict(self):
        if self.update_predcit:
            preds = self.model.predict(self.dataloader)
            if self.model.hist_auc[-1] == self.model.best_auc:
                self.model.best_preds = preds
                self.hist_info[self.model.name] = (self.model.best_auc, self.model.best_preds)

        preds = self.blending_predict()
        return preds

    #@timeit
    def blending_predict(self):
        top_preds = self.get_top_preds()
        ensemble_preds = 0
        for name, rank, auc, weight, preds in top_preds:
            m = np.mean(preds)
            ensemble_preds += weight * preds/m
        return ensemble_preds

    def stacking_predict(self):
        pass

    def softmax(self, x):
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()
