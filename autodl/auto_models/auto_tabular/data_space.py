import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import gc

from autodl.utils.log_utils import log, timeit
from ..auto_tabular.utils.data_utils import ohe2cat
from ..auto_tabular.utils.data_utils import fill_na
from ..auto_tabular.utils.sample import AutoSample
from ..auto_tabular import CONSTANT


class TabularDataSpace:
    def __init__(self, metadata, eda_info, main_df, y, lgb_info):
        self.metadata = metadata

        self.lgb_info = lgb_info

        self.data = main_df
        self.all_idxs = main_df.index
        self.all_train_idxs = main_df.index[main_df.index >= 0]
        self.test_idxs = main_df.index[main_df.index < 0]

        self.y = y

        self.cat_cols = eda_info['cat_cols']
        self.num_cols = eda_info['num_cols']

        self.col2type = {}

        self.post_drop_set = set()

        self.init_col2type()

        self.splits = {}

        self.train_valid_split_idxs()

        self.update = False

    @timeit
    def train_valid_split_idxs(self, ratio=0.2):
        sss = StratifiedShuffleSplit(n_splits=5, test_size=ratio, random_state=0)
        idxs = np.arange(len(self.y))
        i = 0
        for train, val in sss.split(idxs, ohe2cat(self.y)):
            self.splits[i] = (train, val)
            i += 1

        self.train_idxs, self.val_idxs = self.splits[0]

        self.m = len(self.train_idxs)
        self.auto_sample = AutoSample(self.y[self.train_idxs])

    @timeit
    def get_dataloader(self, train_loop_num, round_num, run_num, use_all_data, model_type):

        self.train_idxs, self.val_idxs = self.splits[round_num-1]
        print('round {}'.format(round_num))

        data_loader = {}
        do_sample_col = False

        if use_all_data:
            sample_idxs = self.all_train_idxs
        else:
            sample_idxs = self.train_idxs


        cat_cols = self.get_categories(self.data)

        if model_type == 'nn_keras':
            feats = self.nn_process(self.data, cat_cols)
            feats = pd.DataFrame(feats, index=self.all_idxs)
            data_loader['X'], data_loader['y'] = feats, self.y
            data_loader['shape'] = feats.shape[1]

        elif model_type == 'emb_nn':
            feats = pd.DataFrame(self.data, index=self.all_idxs)
            self.label_encode(feats, cat_cols)
            data_loader['X'], data_loader['y'] = feats, self.y
            data_loader['shape'] = feats.shape[1]
        elif model_type == 'tree':
            feats = self.data
            if do_sample_col:
                data_loader['X'] = feats
            else:
                data_loader['X'] = feats
            data_loader['y'] = self.y
            data_loader['cat_cols'] = cat_cols
        elif model_type == 'lr':
            feats = self.nn_process(self.data, cat_cols)
            feats = pd.DataFrame(feats, index=self.all_idxs)
            data_loader['X'], data_loader['y'] = feats, self.y
            data_loader['cat_cols'] = cat_cols

        data_loader['all_train_idxs'], data_loader['train_idxs'], \
        data_loader['val_idxs'], data_loader['test_idxs'], \
        data_loader['splits'], data_loader['cat_cols']\
            = self.all_train_idxs, sample_idxs, self.val_idxs, self.test_idxs, self.splits, cat_cols

        return data_loader

    def to_tfdataset(self, feats, y=None, mode=None):
        if mode == 'train':
            return tf.data.Dataset.from_tensor_slices((feats, y))
        elif mode == 'val':
            return tf.data.Dataset.from_tensor_slices(feats), y
        elif mode == 'test':
            return tf.data.Dataset.from_tensor_slices(feats)

    def drop_post_drop_column(self, df):
        if len(self.post_drop_set) != 0:
            drop_cols = list(self.post_drop_set)
            df.drop(drop_cols, axis=1, inplace=True)
            gc.collect()

    def get_categories(self, df):
        categories = []
        col_set = set(df.columns)
        for col in self.cat_cols:
            if col in col_set:
                if df[col].nunique() <= 10:
                    categories.append(col)
        return categories

    def update_data(self, df, col2type):
        self.data = df
        self.update_col2type(col2type)

    def update_col2type(self, col2type):
        self.col2type.update(col2type)
        self.type_reset()

    def type_reset(self):

        cat_cols = []
        num_cols = []

        for cname, ctype in self.col2type.items():
            if ctype == CONSTANT.CATEGORY_TYPE:
                cat_cols.append(cname)
            elif ctype == CONSTANT.NUMERICAL_TYPE:
                num_cols.append(cname)

        self.cat_cols = sorted(cat_cols)
        self.num_cols = sorted(num_cols)

    def init_col2type(self):
        for col in self.cat_cols:
            self.col2type[col] = CONSTANT.CATEGORY_TYPE
        for col in self.num_cols:
            self.col2type[col] = CONSTANT.NUMERICAL_TYPE

    def num_fit_transform(self, df, num_cols):
        if not num_cols:
            return []
        scaler = StandardScaler()
        df = df[list(num_cols)]
        arr = df.values
        norm_arr = scaler.fit_transform(arr)
        norm_arr[np.isnan(norm_arr)] = 0
        return norm_arr

    def cat_fit_transform(self, df, cat_cols):
        if len(cat_cols) == 0:
            return []
        res = []
        for cat in cat_cols:
            enc = OneHotEncoder(handle_unknown='ignore')
            arr = enc.fit_transform(df[cat].values.reshape(-1, 1)).toarray()
            res.append(arr)
        cat_feats = np.concatenate(res, axis=1)
        return cat_feats

    def nn_process(self, X, cat_cols):
        num_cols = [col for col in X.columns if col not in cat_cols]


        X = fill_na(X)

        cat_feats = self.cat_fit_transform(X, cat_cols)
        num_feats = self.num_fit_transform(X, num_cols)

        if len(cat_feats) > 0 and len(num_feats) > 0:
            feats = np.concatenate([cat_feats, num_feats], axis=1)
        elif len(cat_feats) > 0:
            feats = cat_feats
        elif len(num_feats) > 0:
            feats = num_feats
        return feats


    def label_encode(self, X, cat_cols):
        if cat_cols == []:
            return

        for col in cat_cols:
            X[col] = pd.Categorical(X[col]).codes

























