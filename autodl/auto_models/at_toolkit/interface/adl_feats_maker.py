#coding:utf-8

class AbsFeatsMaker:
    def __init__(self, feat_tool, feat_name):
        self.feat_tool = feat_tool
        self.feat_name = feat_name
        self.feat_des = None
        self.corr_cls_list = list()

    def make_features(self, raw_data, feats_maker_params:dict):
        raise NotImplementedError
