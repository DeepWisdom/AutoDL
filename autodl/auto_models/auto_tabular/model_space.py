from .model_lib import *


class TabularModelSpace:
    def __init__(self, metadata, info):
        self.metadata = metadata
        self.info = info

        self.model_init_prior = ['lgb', 'cb', 'xgb', 'dnn']
        self.model_prior = self.sort_prior_by_meta()

        self.model_lib = {
            'lr': LogisticRegression,
            'lgb': LGBModel,
            'xgb': XGBModel,
            'cb': CBModel,
            'dnn': DnnModel,
            'enn': ENNModel
        }

    def sort_prior_by_meta(self):
        prior_copy = self.model_init_prior.copy()
        return prior_copy

    def get_model(self, model_name, round_num):
        if model_name in self.model_lib:
            model = self.model_lib[model_name]()
            model.name = '{}_{}'.format(model.name, round_num)
            return model

    def destroy_model(self, model_name):
        if model_name in self.model_lib:
            self.model_prior.remove(model_name)
