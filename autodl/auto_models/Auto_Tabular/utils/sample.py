import numpy as np
import random
import os
from .eda import AutoEDA
from .data_utils import ohe2cat
from .log_utils import info, debug

class AutoSample:
    def __init__(self, y_onehot):
        self.auto_eda = AutoEDA()
        self.y_onehot = y_onehot
        self.sample_num, self.class_num = y_onehot.shape
        self.y_distribution = self.auto_eda.get_label_distribution(y_onehot)
        self.class_idx = self.get_idx_by_class()

    def get_idx_by_class(self):
        """获取每个类的索引"""
        class_idx = {}
        idx = np.arange(self.sample_num)
        for i in range(self.class_num):
            idx_list = idx[self.y_onehot[:,i] == 1]
            class_idx[i] = list(idx_list)
        return class_idx

    def sample_fix_size_by_class(self, per_class_num, max_sample_num=1000, min_sample_num=100):
        """
        对每个类采相同的样本数量
        :param per_class_num: 每个类别的采样数
        :param max_sample_num: 每个类别的最大采样数
        :param min_sample_num: 每个类别的最小采样数
        :return:
        """

        # 要采样的总体 sample_index_id_list.
        final_sample_idx = list()
        min_sample_perclass = int(min_sample_num / self.class_num) + 1

        for label_id in range(self.class_num):
            random_sample_k = per_class_num
            # 如果 k 很小，就用 min_sample_perclass.
            random_sample_k = max(min_sample_perclass, random_sample_k)
            label_idx_list = self.class_idx[label_id]
            label_idx_len = len(label_idx_list)

            # 如果单个类别数量足够，不足两种情况, 避免溢出.
            if label_idx_len > random_sample_k:
                downsample_class_idx = random.sample(population=label_idx_list, k=random_sample_k)
                final_sample_idx.extend(downsample_class_idx)
            else:
                div, mod = divmod(random_sample_k, label_idx_len)
                new_label_idx_list = label_idx_list*div + label_idx_list[:int(mod*label_idx_len)]
                final_sample_idx.extend(new_label_idx_list)

        # 要满足 max_sample_num/min_sample_num. 如果太多，则要重新降低 到上限.
        if len(final_sample_idx) > max_sample_num:
            final_sample_idx = random.sample(population=final_sample_idx, k=max_sample_num)
        # 返回试验Sample的id_list.
        info('final sample length: {}'.format(len(final_sample_idx)))
        random.shuffle(final_sample_idx)
        return final_sample_idx




