# coding:utf-8
import numpy as np
from ...at_speech.data_space.raw_data_space import RawDataNpDb
from ...at_speech.data_space.feats_engine import (
    AbsFeatsMaker,
    KapreMelSpectroGramFeatsMaker,
    LibrosaMelSpectroGramFeatsMaker,
    LbrsTr34FeatsMaker,
)
from ...at_speech.at_speech_cons import *


class FeatsDataDb:
    def __init__(self, raw_train_num, raw_test_num):
        self.raw_train_num = raw_train_num
        self.raw_test_num = raw_test_num
        self.raw_train_feats_data_tables = dict()
        self.raw_test_feats_data_tables = dict()
        self.split_val_feats_data_tables = dict()
        self.raw_feat_makers_table = dict()
        self.split_val_num = None
        self.raw_data_db = RawDataNpDb(self.raw_train_num, self.raw_test_num)

        self.kapre_melspecgram_featmaker = KapreMelSpectroGramFeatsMaker("KAPRE", FEAT_KAPRE_MELSPECGRAM)
        self.lbs_melspecgram_featmaker = LibrosaMelSpectroGramFeatsMaker("LIBROSA", FEAT_LBS_MELSPECGRAM)
        self.lbs_tr34_featmaker = LbrsTr34FeatsMaker("LIBROSA", FEAT_LBS_TR34)

        self.add_feats_data_table(FEAT_KAPRE_MELSPECGRAM, self.kapre_melspecgram_featmaker)
        self.add_feats_data_table(FEAT_LBS_MELSPECGRAM, self.lbs_melspecgram_featmaker)
        self.add_feats_data_table(FEAT_LBS_TR34, self.lbs_tr34_featmaker)

        self.raw_test_feats_status_table = {
            FEAT_KAPRE_MELSPECGRAM: False,
            FEAT_LBS_MELSPECGRAM: False,
            FEAT_LBS_TR34: False,
        }
        self.split_val_feats_status_table = {
            FEAT_KAPRE_MELSPECGRAM: False,
            FEAT_LBS_MELSPECGRAM: False,
            FEAT_LBS_TR34: False,
        }

    def add_feats_data_table(self, feat_name, feats_maker: AbsFeatsMaker):
        if feat_name not in self.raw_train_feats_data_tables.keys():
            self.raw_train_feats_data_tables[feat_name] = np.array([None] * self.raw_train_num)
            self.raw_test_feats_data_tables[feat_name] = np.array([None] * self.raw_test_num)
            self.split_val_feats_data_tables[feat_name] = np.array([None] * self.raw_train_num)
            self.raw_feat_makers_table[feat_name] = feats_maker

    def put_raw_test_feats(self, feat_name, raw_test_feats_np):
        assert feat_name in self.raw_test_feats_data_tables.keys(), "feat_name {} not exists in db".format(feat_name)
        self.raw_test_feats_data_tables[feat_name] = raw_test_feats_np

    def get_raw_test_feats(self, feat_name, feats_maker_params: dict = None, forced=False):
        if self.raw_test_feats_status_table.get(feat_name) is False or forced is True:
            raw_test_feats_np = self.raw_feat_makers_table.get(feat_name).make_features(
                self.raw_data_db.raw_test_x_np_table, feats_maker_params
            )
            self.put_raw_test_feats(feat_name, raw_test_feats_np)
            self.raw_test_feats_status_table[feat_name] = True

        return self.raw_test_feats_data_tables.get(feat_name)

    def get_raw_train_feats(self, feat_name, raw_train_idxs, feats_maker_params: dict = None, forced=False):
        need_make_feats_idxs = list()
        if forced:
            self.raw_train_feats_data_tables[feat_name] = np.array([None] * self.raw_train_num)
            need_make_feats_idxs = raw_train_idxs
        else:
            for raw_train_idx in raw_train_idxs:
                if self.raw_train_feats_data_tables.get(feat_name)[raw_train_idx] is None:
                    need_make_feats_idxs.append(raw_train_idx)

        if len(need_make_feats_idxs) > 0:
            need_make_feats_rawdata = self.raw_data_db.raw_train_x_np_table[need_make_feats_idxs]
            make_feats_done = self.raw_feat_makers_table.get(feat_name).make_features(
                need_make_feats_rawdata, feats_maker_params
            )
            make_feats_done = np.array(make_feats_done)
            for i in range(len(need_make_feats_idxs)):
                self.raw_train_feats_data_tables.get(feat_name)[need_make_feats_idxs[i]] = make_feats_done[i]

        cur_train_feats = [self.raw_train_feats_data_tables.get(feat_name)[i].shape for i in raw_train_idxs]
        return np.stack(self.raw_train_feats_data_tables.get(feat_name)[raw_train_idxs])

    def get_raw_train_y(self, raw_train_idxs):
        return np.stack(self.raw_data_db.raw_train_y_np_table[raw_train_idxs])

    def get_split_val_feats(self, feat_name:str, split_val_idxs:list, feats_maker_params: dict = None, forced=False):
        if self.split_val_num is None:
            self.split_val_num = self.raw_data_db.split_val_sample_num

        if self.split_val_feats_status_table.get(feat_name) is False or forced is True:
            self.split_val_feats_data_tables[feat_name] = np.array([None] * self.split_val_num)
            need_make_feats_rawdata = self.raw_data_db.raw_train_x_np_table[split_val_idxs]
            make_feats_done = self.raw_feat_makers_table.get(feat_name).make_features(
                need_make_feats_rawdata, feats_maker_params
            )
            make_feats_done = np.array(make_feats_done)
            self.split_val_feats_data_tables[feat_name] = make_feats_done
            self.split_val_feats_status_table[feat_name] = True

        return self.split_val_feats_data_tables.get(feat_name)

