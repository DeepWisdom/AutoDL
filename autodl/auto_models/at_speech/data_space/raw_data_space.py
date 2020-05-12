# coding:utf-8
import numpy as np


class RawDataNpDb:
    def __init__(self, train_mum, test_num):
        self.train_num = train_mum
        self.test_num = test_num
        self.raw_train_x_np_table = np.array([None] * self.train_num)
        self.raw_train_y_np_table = np.array([None] * self.train_num)
        self.raw_train_x_np_table_filled = None
        self.raw_train_y_np_table_filled = None
        self.raw_test_x_np_table = np.array([None] * self.test_num)
        self.raw_train_np_filled_num = 0
        self.if_raw_test_2_np_done = False
        self.split_val_x_np_table = None
        self.split_val_y_np_table = None
        self.split_val_sample_idxs = list()
        self.split_val_sample_num = None

    def put_raw_train_np(self, raw_train_x_array, raw_train_y_array):
        put_len = len(raw_train_x_array)
        for i in range(put_len):
            self.raw_train_x_np_table[i] = np.array(raw_train_x_array[i])
            self.raw_train_y_np_table[i] = np.array(raw_train_y_array[i])

        self.raw_train_np_filled_num = put_len
        self.raw_train_x_np_table_filled = self.raw_train_x_np_table[: self.raw_train_np_filled_num]
        self.raw_train_y_np_table_filled = self.raw_train_y_np_table[: self.raw_train_np_filled_num]

    def put_raw_test_np(self, raw_test_x_array):
        self.raw_test_x_np_table = raw_test_x_array.copy()
        self.if_raw_test_2_np_done = True

    def put_split_valid_np(self, val_sample_idxs:list):
        self.split_val_sample_idxs = val_sample_idxs
        self.split_val_x_np_table = self.raw_train_x_np_table[self.split_val_sample_idxs]
        self.split_val_y_np_table = self.raw_train_y_np_table[self.split_val_sample_idxs]
        self.split_val_sample_num = len(val_sample_idxs)
