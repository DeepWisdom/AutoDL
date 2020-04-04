import numpy as np
import pandas as pd
from .feat_namer import FeatNamer

from sklearn.model_selection import StratifiedKFold
from Auto_Tabular import CONSTANT
from joblib import Parallel, delayed
from Auto_Tabular.utils.log_utils import log ,timeit


class TargetEncoder:
    def __init__(self, cols):
        self.cats = cols
        self.map = {}
        self.means = {}
        self.a = 1
        self.num_class = None
        self.y_df = None

    def fit(self, X, y):
        self.num_class = y.shape[1]
        self.y_df = pd.DataFrame(y, columns=range(self.num_class), index=X.index).astype(float)

        for i in range(self.num_class):
            y_ss = y[i]
            self.means[i] = y_ss.mean()

        for col in self.cats:
            for i in range(self.num_class):
                y_ss = self.y_df[i]
                result = y_ss.groupby(X[col]).agg(['sum', 'count'])
                self.map[(col, i)] = result

    def transform(self, X, y=None):

        for (col, i), colmap in self.map.items():
            if y is None:
                level_notunique = colmap['count'] > 1
                level_means = ((colmap['sum'] + self.means[i]) / (colmap['count'] + self.a)).where(level_notunique, self.means[i])
                ss = X[col].map(level_means)
                col_name = 'n_target_encode_{}_{}'.format(col, i)
                X[col_name] = ss
            else:
                y_ss = self.y_df[i]
                temp = y_ss.groupby(X[col].astype(str)).agg(['cumsum', 'cumcount'])
                ss = (temp['cumsum'] - y_ss + self.means[i]) / (temp['cumcount'] + self.a)
                col_name = 'n_target_encode_{}_{}'.format(col, i)
                X[col_name] = ss

    def fit_transform(self, X, y):
        self.fit(X, y)
        self.transform(X, y)


class MyTargetEncoder:
    def __init__(self, cols):
        self.cats = cols
        self.map = {}
        self.means = {}
        self.a = 1
        self.num_class = None
        self.y_df = None
        self.folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    def fit(self, X, y):
        self.num_class = y.shape[1]
        self.y_df = pd.DataFrame(y, columns=range(self.num_class), index=X.index).astype(float)

        for i in range(self.num_class):
            y_ss = y[i]
            self.means[i] = y_ss.mean()

        for col in self.cats:
            for i in range(self.num_class):
                y_ss = self.y_df[i]
                result = y_ss.groupby(X[col]).agg(['sum', 'count'])
                self.map[(col, i)] = result

    def transform(self, X, y=None):

        for (col, i), colmap in self.map.items():
            if y is None:
                level_notunique = colmap['count'] > 1
                level_means = ((colmap['sum'] + self.means[i]) / (colmap['count'] + self.a)).where(level_notunique, self.means[i])
                ss = X[col].map(level_means)
                col_name = 'n_target_encode_{}_{}'.format(col, i)
                X[col_name] = ss
            else:
                y_ss = self.y_df[i]
                temp = y_ss.groupby(X[col].astype(str)).agg(['cumsum', 'cumcount'])
                ss = (temp['cumsum'] - y_ss + self.means[i]) / (temp['cumcount'] + self.a)
                col_name = 'n_target_encode_{}_{}'.format(col, i)
                X[col_name] = ss

    def fit_transform(self, X, y):
        self.fit(X, y)
        self.transform(X, y)



class GroupbyMeanMinusSelf:


    def fit(self, X, y, cat_cols, num_cols):
        pass

    def transform(self, X, y, cat_cols, num_cols):
        num_cols = [i for i in num_cols if i.startswith('n_')]

        if cat_cols == [] or num_cols == []:
            return


        def groupby_mean(df):
            cat_col, num_col = df.columns[0], df.columns[1]

            means = df.groupby(cat_col, sort=False)[num_col].mean()
            ss1 = df[cat_col].map(means)

            param = 'mean'
            obj= '({})({})'.format(cat_col, num_col)
            ss1.name = FeatNamer.gen_feat_name(self.__class__.__name__, obj, param, CONSTANT.NUMERICAL_TYPE)

            ss2 = ss1 - df[num_col]
            param = 'minus'
            ss2.name = FeatNamer.gen_feat_name(self.__class__.__name__, obj, param, CONSTANT.NUMERICAL_TYPE)

            return ss1, ss2

        exec_cols = []
        for col1 in cat_cols:
            for col2 in num_cols:
                exec_cols.append((col1, col2))

        opt = groupby_mean
        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(
            delayed(opt)(X[[col1, col2]]) for col1, col2 in exec_cols)
        if res:
            for tp in res:
                for ss in tp:
                    X[ss.name] = ss

    @timeit
    def fit_transform(self, X, y, cat_cols, num_cols):
        self.fit(X, y, cat_cols, num_cols)
        self.transform(X, y, cat_cols, num_cols)











