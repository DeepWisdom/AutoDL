import numpy as np
import random
from sklearn.model_selection import train_test_split

# from at_toolkit import info, error

def ohe2cat(label):
    return np.argmax(label, axis=1)

def ohe2cat_new(y_labels):
    assert isinstance(y_labels, np.ndarray)
    one_index = np.argwhere(y_labels == 1)
    sample_num = y_labels.shape[0]
    cat_list = [-1] * sample_num
    for index_item in one_index:
        cat_list[index_item[0]] = index_item[1]

    return cat_list


class AutoSamplerBasic:
    def __init__(self, class_num):
        self._num_classes = class_num
        self._max_class_num = 0
        self._min_class_num = 0
        self.train_y = None
        self._train_y_num = None

    def init_train_y(self, train_y):
        self.train_y = train_y
        self._train_y_num = len(self.train_y)

    def init_each_class_index_by_y(self, cur_train_y):
        cur_train_y = np.stack(cur_train_y)
        each_class_count = np.sum(np.array(cur_train_y), axis=0)
        self._max_class_num, self._min_class_num = int(np.max(each_class_count)), int(np.min(each_class_count))

        each_class_index = []
        for i in range(self._num_classes):
            each_class_index.append(
                list(np.where(cur_train_y[:, i] == 1)[0]))

        return each_class_index

    def init_even_class_index_by_each(self, each_class_index_list):
        even_class_index = []
        sample_per_class = max(int(self._train_y_num / self._num_classes), 1)
        for i in range(self._num_classes):
            class_cnt = len(each_class_index_list[i])
            tmp = []
            if class_cnt == 0:
                pass
            elif class_cnt < sample_per_class:
                tmp = each_class_index_list[i] * \
                      int(sample_per_class / class_cnt)
                tmp += random.sample(
                    each_class_index_list[i],
                    sample_per_class - len(tmp))
            else:
                tmp += random.sample(
                    each_class_index_list[i], sample_per_class)
            random.shuffle(tmp)
            even_class_index.append(tmp)

        return even_class_index


class AutoSpeechEDA(object):
    def __init__(self, data_expe_flag=False, x_train=None, y_train=None):
        self.train_data_sinffer_num_per_class = 50
        self.expe_x_data_num = 3
        self.expe_y_labels_num = 3
        self.data_expe_flag = data_expe_flag

    def get_y_label_distribution_by_bincount(self, y_onehot_labels):
        y_sample_num, y_label_num = y_onehot_labels.shape
        y_as_label = ohe2cat(y_onehot_labels)
        y_as_label_bincount = np.bincount(y_as_label)
        y_label_distribution_array = y_as_label_bincount / y_sample_num
        y_label_distribution_array = list(y_label_distribution_array)
        return y_label_distribution_array

    def get_y_label_eda_report(self, y_onehot_labels):
        y_train_len_num = len(y_onehot_labels)
        if self.data_expe_flag:
            expe_x_data_list = [a_x_data.tolist() for a_x_data in y_onehot_labels[:self.expe_x_data_num]]
        y_sample_num, y_label_num = y_onehot_labels.shape
        y_label_distribution_array = self.get_y_label_distribution_by_bincount(y_onehot_labels=y_onehot_labels)
        eda_y_report = dict()
        eda_y_report["y_sample_num"] = int(y_sample_num)
        eda_y_report["y_class_num"] = int(y_label_num)
        eda_y_report["y_label_distribution_array"] = y_label_distribution_array
        return eda_y_report

    def get_x_data_report(self, x_data):
        x_sample_num = len(x_data)
        if self.data_expe_flag:
            expe_x_data_list = [a_x_data.tolist() for a_x_data in x_data[:self.expe_x_data_num]]
        x_train_word_len_list = list()
        for x_train_sample in x_data:
            len_a_x_sample = x_train_sample.shape[0]
            x_train_word_len_list.append(len_a_x_sample)
        x_train_word_len_array = np.array(x_train_word_len_list)
        x_train_sample_mean = x_train_word_len_array.mean()
        x_train_sample_std = x_train_word_len_array.std()

        eda_x_data_report = dict()
        eda_x_data_report["x_total_seq_num"] = int(x_train_word_len_array.sum())
        eda_x_data_report["x_seq_len_mean"] = int(x_train_sample_mean)
        eda_x_data_report["x_seq_len_std"] = x_train_sample_std
        eda_x_data_report["x_seq_len_max"] = int(x_train_word_len_array.max())
        eda_x_data_report["x_seq_len_min"] = int(x_train_word_len_array.min())
        eda_x_data_report["x_seq_len_median"] = int(np.median(x_train_word_len_array))
        eda_x_data_report["x_sample_num"] = int(x_sample_num)
        return eda_x_data_report


class AutoSpSamplerNew(object):
    def __init__(self, y_train_labels):

        self.autosp_eda = AutoSpeechEDA()
        self.y_train_labels = y_train_labels
        self.y_train_cat_list = None
        self.y_class_num = None
        self.g_label_sample_id_list = list()

    def set_up(self):

        self.y_train_cat_list = ohe2cat(self.y_train_labels)
        y_labels_eda_report = self.autosp_eda.get_y_label_eda_report(y_onehot_labels=self.y_train_labels)
        self.y_class_num = y_labels_eda_report.get("y_class_num")
        for y_label_id in range(self.y_class_num):
            label_sample_id_list = list(np.where(self.y_train_cat_list == y_label_id)[0])
            self.g_label_sample_id_list.append(label_sample_id_list)

    def get_downsample_index_list_by_class(self, per_class_num, max_sample_num, min_sample_num):
        train_data_sample_id_list = list()
        min_sample_perclass = int(min_sample_num / self.y_class_num)
        for y_label_id in range(self.y_class_num):
            random_sample_k = per_class_num
            random_sample_k = max(min_sample_perclass, random_sample_k)
            label_sample_id_list = self.g_label_sample_id_list[y_label_id]
            if len(label_sample_id_list) > random_sample_k:
                downsampling_label_sample_id_list = random.sample(population=label_sample_id_list, k=random_sample_k)
                train_data_sample_id_list.extend(downsampling_label_sample_id_list)
            else:
                train_data_sample_id_list.extend(label_sample_id_list)

        if len(train_data_sample_id_list) > max_sample_num:
            train_data_sample_id_list = random.sample(population=train_data_sample_id_list, k=max_sample_num)
        return train_data_sample_id_list


    def get_downsample_index_list_by_random(self, max_sample_num, min_sample_num):

        assert min_sample_num <= max_sample_num, "Error: min_sample_num={}, max_sample_num={}".format(min_sample_num, max_sample_num)

        sample_num, class_num = self.y_train_labels.shape[0], self.y_train_labels.shape[1]
        if sample_num <= min_sample_num:
            cur_sample_idxs = list(range(sample_num))
            random.shuffle(cur_sample_idxs)
        elif min_sample_num < sample_num <= max_sample_num:
            cur_sample_idxs = list(range(sample_num))
            random.shuffle(cur_sample_idxs)
        elif sample_num > max_sample_num:
            cur_sample_idxs = random.sample(population=range(sample_num), k=max_sample_num)
        else:
            cur_sample_idxs = list(range(sample_num))
            random.shuffle(cur_sample_idxs)

        return cur_sample_idxs


class AutoValidSplitor:
    def __init__(self, class_num):
        self.class_num = class_num
        self.ohe_y_labels = None

    def get_valid_sample_idxs(self, ohe_y_labels, val_num, mode="random"):
        self.ohe_y_labels = ohe_y_labels
        assert self.class_num == ohe_y_labels.shape[1]
        y_sample_num = ohe_y_labels.shape[0]
        if mode == "random":
            split_res = train_test_split(range(y_sample_num), test_size=val_num, shuffle=True)
            return split_res[1]
        else:
            y_cats_list = ohe2cat(ohe_y_labels)
            y_label_inverted_index_array = list()
            for y_label_id in range(self.class_num):
                label_sample_id_list = list(np.where(y_cats_list == y_label_id)[0])
                y_label_inverted_index_array.append(label_sample_id_list)

            val_sample_idxs = list()
            avg_num_perclass = int(np.ceil(val_num/self.class_num))
            for y_label_id in range(self.class_num):
                a_label_sample_idxs = y_label_inverted_index_array[y_label_id]
                if len(a_label_sample_idxs) >= avg_num_perclass:
                    sampled_sampled_idxs = random.sample(population=a_label_sample_idxs,
                                                                      k=avg_num_perclass)
                else:
                    sampled_sampled_idxs = a_label_sample_idxs

                val_sample_idxs.extend(sampled_sampled_idxs)
            return val_sample_idxs

    def get_val_label_array(self, sampld_idxs):
        val_label_array = list()
        for sampld_idx in sampld_idxs:
            i, = np.where(self.ohe_y_labels[sampld_idx] == 1)

            val_label_array.append(i)

        return val_label_array


def minisamples_edaer(mini_xs:list, mini_y:np.ndarray):

    x_seq_len_list = list()
    for x_train_sample in mini_xs:
        len_a_x_sample = x_train_sample.shape[0]
        x_seq_len_list.append(len_a_x_sample)
    x_word_len_array = np.array(x_seq_len_list)
    x_seq_len_mean = x_word_len_array.mean()
    x_seq_len_std = x_word_len_array.std()

    mini_num, class_num = mini_y.shape[0], mini_y.shape[1]
    each_class_index = []
    class_val_count = 0
    for i in range(class_num):
        where_i = np.where(mini_y[:, i] == 1)
        class_i_ids = list(where_i[0])
        if len(class_i_ids) > 0:
            class_val_count += 1

        each_class_index.append(class_i_ids)

    class_cover_rate = round(class_val_count/class_num, 4)
    class_dis_array = [round(len(i)/mini_num, 4) for i in each_class_index]

    onehot_y_sum = np.sum(mini_y)
    is_multilabel = False
    if onehot_y_sum > mini_num:
        is_multilabel = True

    mini_eda_report = {
        "minis_num": len(mini_xs),
        "x_seqlen_mean": round(x_seq_len_mean, 4),
        "x_seqlen_std": round(x_seq_len_std, 4),
        "y_cover_rate": class_cover_rate,
        "y_dis_array": class_dis_array,
        "is_multilabel": is_multilabel
    }
    return mini_eda_report


def sample_y_edaer(samples_y: np.ndarray):
    mini_num, class_num = samples_y.shape[0], samples_y.shape[1]
    each_class_index = []
    class_val_count = 0
    for i in range(class_num):
        where_i = np.where(samples_y[:, i] == 1)
        class_i_ids = list(where_i[0])
        if len(class_i_ids) > 0:
            class_val_count += 1

        each_class_index.append(class_i_ids)

    class_cover_rate = round(class_val_count / class_num, 4)
    class_dis_array = [round(len(i) / mini_num, 4) for i in each_class_index]
    samples_y_eda_report = {
        "y_cover_rate": class_cover_rate,
        "y_dis_array": class_dis_array
    }
    return samples_y_eda_report


