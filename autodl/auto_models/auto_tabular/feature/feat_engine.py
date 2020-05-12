from .feat_gen import *
from sklearn.utils import shuffle
from autodl.utils.log_utils import timeit


class FeatEngine:
    def __init__(self):

        self.order2s = []

    def fit(self, data_space, order):
        if order != 2:
            return
        order_name = 'order{}s'.format(order)
        pipline = getattr(self, order_name)
        self.feats = []
        for feat_cls in pipline:
            feat = feat_cls()
            feat.fit(data_space)
            self.feats.append(feat)

    def transform(self, data_space, order):
        for feat in self.feats:
            feat.transform(data_space)

    @timeit
    def fit_transform(self, data_space, order, info=None):
        if order != 2:
            return
        order_name = 'order{}s'.format(order)
        pipline = getattr(self, order_name)
        X, y = data_space.data, data_space.y
        cats = data_space.cat_cols
        for feat_cls in pipline:
            feat = feat_cls()
            feat.fit_transform(X, y, cat_cols=cats, num_cols=info['imp_nums'])
        data_space.data = X
        data_space.update = True
