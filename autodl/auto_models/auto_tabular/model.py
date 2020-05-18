#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc     : tabular train&test model

from tensorflow.python.client import device_lib
import logging
import numpy as np
import os
import sys
import pandas as pd
import gc
import lightgbm as lgb

from .explore import Explore
from .data_space import TabularDataSpace
from .model_space import TabularModelSpace
from .utils.eda import AutoEDA
from .utils.data_utils import ohe2cat
from ..auto_tabular import CONSTANT
import random
import time
from catboost import CatBoostClassifier

import tensorflow as tf
from tensorflow.python.keras.backend import set_session

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
config = tf.ConfigProto()
# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True
# to log device placement (on which device the operation ran)
config.log_device_placement = False
# (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
# set this TensorFlow session as the default session for Keras
set_session(sess)


class Model(object):

    def __init__(self, metadata):
        """
        Args:
            metadata: an AutoDLMetadata object. Its definition can be found in
                    AutoDL_ingestion_program/dataset.py
        """
        self.done_training = False
        self.metadata = metadata

        self.metadata_info = metadata.metadata_
        self.train_loop_num = 0

        self.auto_eda = AutoEDA()

        self.X = []
        self.Y = []

        self.pre_increament_preds = True

        self.X_test = None

        self.next_element = None

        self.lgb_info = {}

        self.imp_cols = None

        self.is_multi_label = None

        self.models = {}

        self.sample_cols = None

        self.unknow_cols = None

        self.first_preds = False

        self.model = None

        self.keep_training_booster = False

    # @timeit
    def fit(self, dataset, remaining_time_budget=None):
        """Train this algorithm on the tensorflow |dataset|.

        This method will be called REPEATEDLY during the whole training/predicting
        process. So your `train` method should be able to handle repeated calls and
        hopefully improve your model performance after each call.

        ****************************************************************************
        ****************************************************************************
        IMPORTANT: the loop of calling `train` and `test` will only run if
                self.done_training = False
            (the corresponding code can be found in ingestion.py, search
            'M.done_training')
            Otherwise, the loop will go on until the time budget is used up. Please
            pay attention to set self.done_training = True when you think the model is
            converged or when there is not enough time for next round of training.
        ****************************************************************************
        ****************************************************************************

        Args:
            dataset: a `tf.data.Dataset` object. Each of its examples is of the form
                        (example, labels)
                    where `example` is a dense 4-D Tensor of shape
                        (sequence_size, row_count, col_count, num_channels)
                    and `labels` is a 1-D Tensor of shape
                        (output_dim,).
                    Here `output_dim` represents number of classes of this
                    multilabel classification task.

                    IMPORTANT: some of the dimensions of `example` might be `None`,
                    which means the shape on this dimension might be variable. In this
                    case, some preprocessing technique should be applied in order to
                    feed the training of a neural network. For example, if an image
                    dataset has `example` of shape
                        (1, None, None, 3)
                    then the images in this datasets may have different sizes. On could
                    apply resizing, cropping or padding in order to have a fixed size
                    input tensor.

            remaining_time_budget: time remaining to execute train(). The method
                    should keep track of its execution time to avoid exceeding its time
                    budget. If remaining_time_budget is None, no time budget is imposed.
        """
        self.train_loop_num += 1
        if self.pre_increament_preds:
            self.X_train, self.Y_train = self.to_numpy_train(dataset)
            self.X_train = pd.DataFrame(self.X_train)

        if not self.pre_increament_preds and self.train_loop_num > 50:
            self.done_training = True

    # @timeit
    def predict(self, dataset, remaining_time_budget=None):
        """Make predictions on the test set `dataset` (which is different from that
        of the method `train`).

        Args:
            Same as that of `train` method, except that the `labels` will be empty
                    since this time `dataset` is a test set.
        Returns:
            predictions: A `numpy.ndarray` matrix of shape (sample_count, output_dim).
                    here `sample_count` is the number of examples in this dataset as test
                    set and `output_dim` is the number of labels to be predicted. The
                    values should be binary or in the interval [0,1].
        """
        if self.pre_increament_preds or self.first_preds:
            if self.X_test is None:
                self.X_test, _ = self.to_numpy_test(dataset)
                self.X_test = pd.DataFrame(self.X_test)

            preds = self.simple_lgb(self.X_train, self.Y_train, self.X_test)
            if self.first_preds:
                self.first_preds = False
                self.train_loop_num = 0
        else:
            if self.train_loop_num == 1:
                self.X_test.index = -self.X_test.index - 1
                main_df = pd.concat([self.X_train, self.X_test], axis=0)

                self.X_test.drop(self.X_test.columns, axis=1, inplace=True)
                self.X_train.drop(self.X_train.columns, axis=1, inplace=True)
                del self.X_train, self.X_test, self.X, self.Y
                gc.collect()

                eda_info = self.auto_eda.get_info(main_df)
                eda_info['is_multi_label'] = self.is_multi_label
                self.data_space = TabularDataSpace(self.metadata_info, eda_info, main_df, self.Y_train, self.lgb_info)
                self.model_space = TabularModelSpace(self.metadata_info, eda_info)
                self.explore = Explore(self.metadata_info, eda_info, self.model_space, self.data_space)
            print('time', remaining_time_budget)
            self.explore.explore_space(train_loop_num=self.train_loop_num, time_remain=remaining_time_budget)
            preds = self.explore.predict()
        return preds

    # @timeit
    def simple_lgb(self, X, y, test_x):
        self.params = {
            "boosting_type": "gbdt",
            "objective": "multiclass",
            'num_class': y.shape[1],
            "metric": "multi_logloss",
            "verbosity": -1,
            "seed": CONSTANT.SEED,
            "num_threads": CONSTANT.JOBS,
        }

        self.hyperparams = {
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'max_bin': 110,
            'subsample': 1,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'min_child_weight': 0.001,
            'min_split_gain': 0.02,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            "learning_rate": 0.1,
            'num_boost_round': 10,
        }

        print('sample lgb predict num:', self.train_loop_num)
        if self.train_loop_num == 1:
            if X.shape[1] > 500:
                self.sample_cols = list(set(X.columns))[::2]
                self.unknow_cols = [col for col in X.columns if col not in self.sample_cols]
                X = X[self.sample_cols]
                test_x = test_x[self.sample_cols]
            if self.is_multi_label:
                self.params['num_class'] = 2
                all_preds = []
                for cls in range(y.shape[1]):
                    cls_y = y[:, cls]
                    data = lgb.Dataset(X, cls_y)
                    self.models[cls] = lgb.train({**self.params, **self.hyperparams}, data)
                    preds = self.models[cls].predict(test_x)
                    all_preds.append(preds[:, 1])
                preds = np.stack(all_preds, axis=1)
            else:
                lgb_train = lgb.Dataset(X, ohe2cat(y))
                self.model = lgb.train({**self.params, **self.hyperparams}, train_set=lgb_train)
                preds = self.model.predict(test_x)
            self.log_feat_importances()
        else:
            self.hyperparams['num_boost_round'] += self.train_loop_num * 5
            self.hyperparams['num_boost_round'] = min(40, self.hyperparams['num_boost_round'])
            print(self.hyperparams['num_boost_round'])

            if self.is_multi_label:
                models = {}
                all_preds = []
                for cls in range(y.shape[1]):
                    cls_y = y[:, cls]
                    data = lgb.Dataset(X[self.imp_cols], cls_y)
                    models[cls] = lgb.train({**self.params, **self.hyperparams}, data)
                    preds = models[cls].predict(test_x[self.imp_cols])
                    all_preds.append(preds[:, 1])
                preds = np.stack(all_preds, axis=1)
            else:
                lgb_train = lgb.Dataset(X[self.imp_cols], ohe2cat(y))
                model = lgb.train({**self.params, **self.hyperparams}, train_set=lgb_train)
                preds = model.predict(test_x[self.imp_cols])
        return preds

    # @timeit
    def to_numpy_train(self, dataset):
        if self.next_element is None:
            dataset = dataset.batch(100)
            iterator = dataset.make_one_shot_iterator()
            self.next_element = iterator.get_next()
        if self.train_loop_num == 1 or self.train_loop_num == 2:
            size = 500
            #1000
        elif self.train_loop_num == 3 or self.train_loop_num == 4:
            size = 1000
        else:
            size = 500*2**(self.train_loop_num-3)
        for i in range(int(size/100)):
            try:
                example, labels = sess.run(self.next_element)
                self.X.extend(example)
                self.Y.extend(labels)
            except tf.errors.OutOfRangeError:
                self.pre_increament_preds = False
                if self.train_loop_num == 1:
                    self.first_preds = True
                self.train_loop_num = 1
                break

        X, y = np.asarray(self.X), np.asarray(self.Y)

        if self.train_loop_num == 1:
            if any(y.sum(axis=1) > 1):
                print('is multi label')
                self.is_multi_label = True

        return X[:, 0, 0, :, 0], y

    def to_numpy_test(self, dataset):
        dataset = dataset.batch(100)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        X = []
        Y = []
        while True:
            try:
                example, labels = sess.run(next_element)
                X.extend(example)
                Y.extend(labels)
            except tf.errors.OutOfRangeError as exp:
                break
        X, y = np.asarray(X), np.asarray(Y)
        return X[:, 0, 0, :, 0], y

    def log_feat_importances(self):
        if not self.is_multi_label:
            importances = pd.DataFrame({'features': [i for i in self.model.feature_name()],
                                                                    'importances': self.model.feature_importance("gain")})
        else:
            importances = pd.DataFrame({'features': [i for i in self.models[0].feature_name()],
                                                                    'importances': self.models[0].feature_importance("gain")})

        importances.sort_values('importances', ascending=False, inplace=True)

        importances = importances[importances['importances'] > 0]
        size = int(len(importances)*0.8)
        if self.imp_cols is None:
            if self.unknow_cols is not None:
                self.imp_cols = self.unknow_cols + [int(col) for col in importances['features'].values]
            else:
                self.imp_cols = [int(col) for col in importances['features'].values]
        else:
            self.imp_cols = [int(col) for col in importances['features'].values]
        self.lgb_info['imp_cols'] = self.imp_cols

    def infer_domain(self):
        """Infer the domain from the shape of the 4-D tensor."""
        row_count, col_count = self.metadata.get_matrix_size(0)
        sequence_size = self.metadata.get_sequence_size()
        channel_to_index_map = dict(self.metadata.get_channel_to_index_map())
        domain = None
        if sequence_size == 1:
            if row_count == 1 or col_count == 1:
                domain = "tabular"
            else:
                domain = "image"
        else:
            if row_count == 1 and col_count == 1:
                if channel_to_index_map:
                    domain = "text"
                else:
                    domain = "speech"
            else:
                domain = "video"
        self.domain = domain
        tf.logging.info("The inferred domain of the dataset is: {}.".format(domain))
        return domain


def has_regular_shape(dataset):
    """Check if the examples of a TF dataset has regular shape."""
    with tf.Graph().as_default():
        iterator = dataset.make_one_shot_iterator()
        example, labels = iterator.get_next()
        return all([x > 0 for x in example.shape])
