"""
MIT License

Copyright (c) 2019 Lenovo Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import string
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing import text

CHI_WORD_LENGTH = 2
MAX_CHAR_LENGTH = 96
MAX_VOCAB_SIZE = 20000
MAX_SEQ_LENGTH = 601

MAX_VALID_PERCLASS_SAMPLE = 400
MAX_SAMPLE_TRIAN = 18000

MAX_TRAIN_PERCLASS_SAMPLE = 1000
min_token_length = 2

MAX_EN_CHAR_LENGTH = 35

from autodl.utils.log_utils import info
from ...at_nlp.data_manager.sample_utils import *
from ...at_nlp.utils import color_msg

punctuations = string.punctuation


def sample_input_data(input_x, input_y, num_classes, max_num=500):
    all_index = []
    meta_train_index = []

    for i in range(num_classes):
        all_index.append(
            list(np.where((input_y[:, i] == 1) == True)[0]))

    for i in range(num_classes):  # 按label类别抽取
        if len(all_index[i]) < max_num and len(all_index[i]) > 0:
            tmp = all_index[i] * int(
                max_num / len(all_index[i]))
            tmp += random.sample(all_index[i],
                                 max_num - len(tmp))
            meta_train_index += tmp
        else:
            meta_train_index += random.sample(
                all_index[i], max_num)

    random.shuffle(meta_train_index)

    train_sample_x = [input_x[i] for i in meta_train_index]
    train_sample_y = input_y[meta_train_index, :]
    return train_sample_x, train_sample_y


class DataGenerator(object):
    def __init__(self,
                 x_train, y_train,
                 metadata,
                 imbalance_level=-1,
                 multi_label=False):

        self.meta_data_x, self.meta_data_y = x_train, y_train
        # 第一次增量的数据就是初始的数据
        self.update_x, self.update_y = x_train, y_train

        self.metadata = metadata

        self.num_classes = self.metadata['class_num']
        self.num_samples_train = self.metadata['train_num']
        self.language = metadata['language']
        self.multi_label = multi_label

        self.val_index = None
        self.tokenizer = None
        self.max_length = None
        self.sample_num_per_class = None
        self.data_feature = {}
        self.eda_feature = {}
        self.pseudo_x_train_size = 0
        self.full_x = []
        self.full_y = np.array([])

        self.x_dict = {i: [] for i in range(self.num_classes)}
        self.imbalance_flg = False
        self.do_generate_sample = False
        self.empty_class_ = []
        self.meta_train_x = []
        self.meta_train_y = np.array([])

        self.full_index = None
        self.imbalance_level = imbalance_level
        self.MAX_TRAIN_PERCLASS_SAMPLE = MAX_TRAIN_PERCLASS_SAMPLE

        if self.num_classes <= 5 and self.imbalance_level <= 1 and self.num_classes > 2:
            self.MAX_TRAIN_PERCLASS_SAMPLE = 3000
        elif self.num_classes == 2 and self.imbalance_level <= 1:
            self.MAX_TRAIN_PERCLASS_SAMPLE = 3500

        if self.multi_label:
            if self.num_classes<50: #类别少，每类取100条
                self.MAX_TRAIN_PERCLASS_SAMPLE = 100
            elif self.num_classes<100:
                self.MAX_TRAIN_PERCLASS_SAMPLE = 50
            else:
                self.MAX_TRAIN_PERCLASS_SAMPLE = 20

    def update_meta_data(self, x_train, y_train):
        self.update_x = x_train
        self.update_y = y_train
        # 添加新增数据
        self.meta_data_x = self.meta_data_x + x_train
        self.meta_data_y = np.concatenate([self.meta_data_y, y_train], axis=0)

    def set_sample_num_per_class(self, sample_num_per_class):
        self.sample_num_per_class = sample_num_per_class

    def _get_val_index(self, train_label_ratio, train_num, index):
        # 设置采样比例
        val_ratio = set_val_ratio(train_label_ratio, train_num)
        # 随机采样
        tmp = random.sample(index, int(len(index) * val_ratio))
        # 设置采样最大值
        if len(tmp) > MAX_VALID_PERCLASS_SAMPLE:
            tmp = tmp[:MAX_VALID_PERCLASS_SAMPLE]
        return tmp

    def sample_val_index(self, input_y):
        train_label_distribution = np.sum(np.array(input_y), 0)
        train_label_ratio = train_label_distribution / np.sum(train_label_distribution)
        all_index = get_sample_index(input_y, self.num_classes)
        val_index = []

        for i in range(self.num_classes):
            if train_label_distribution[i] == 0:
                continue
            tmp = self._get_val_index(train_label_ratio=train_label_ratio[i],
                                      train_num=train_label_distribution[i],
                                      index=all_index[i])
            val_index += tmp
            # 去除train/val 交叉样本
            all_index[i] = list(set(all_index[i]).difference(set(tmp)))

        return all_index, val_index

    def _set_max_train_sample_num(self, train_label_distribution):
        self.max_sample_num_per_class = int(
            np.max(train_label_distribution) * 4 / 5)

        if self.sample_num_per_class is None:
            if self.num_samples_train < MAX_SAMPLE_TRIAN:
                self.sample_num_per_class = self.max_sample_num_per_class
            else:
                self.sample_num_per_class = min(self.max_sample_num_per_class, self.MAX_TRAIN_PERCLASS_SAMPLE)
        else:
            # 避免类别数多的情况下，第一次进样少，导致后面连续采样过低
            self.sample_num_per_class = max(self.max_sample_num_per_class, int(np.mean(train_label_distribution)))

        if self.imbalance_flg:
            max_sample_num = min(self.sample_num_per_class, int(np.mean(train_label_distribution)))
            max_sample_num = min(max_sample_num, self.MAX_TRAIN_PERCLASS_SAMPLE)
        else:
            max_sample_num = min(self.sample_num_per_class, self.MAX_TRAIN_PERCLASS_SAMPLE)

        return max_sample_num

    def balance_sampling_index(self, all_index, train_label_distribution):
        max_sample_num = self._set_max_train_sample_num(train_label_distribution)
        meta_train_index = balance_sampling(all_index, max_sample_num, num_classes=self.num_classes)
        random.shuffle(meta_train_index)
        self.meta_train_index = meta_train_index
        return meta_train_index

    def check_imbalance_level(self, train_label_distribution):
        if self.normal_std >= 0.1 or 0.0 in train_label_distribution:
            self.imbalance_flg = True

    def generate_presudo_samples(self, all_index):
        new_generate_index = []
        for i in range(self.num_classes):
            new_generate_index.append([])

        for i in range(self.num_classes):  # 按label类别抽取
            if len(all_index[i]) == 0:
                self.do_generate_sample = True
                new_samples = do_random_generate_sample(data_x=self.meta_data_x, language=self.language, num=1)
                new_generate_index[i] = new_samples

        return new_generate_index

    def generate_pseudo_samples(self, train_x, train_y):
        info("Do Radam Create Samples!")
        for i in range(self.num_classes):
            new_samples = self.new_generate_samples_idx[i]
            if len(new_samples) == 0:
                continue
            train_x.extend(new_samples)
            new_label = np.zeros((len(new_samples), self.num_classes))
            new_label[:, i] = 1
            train_y = np.concatenate([train_y, new_label], axis=0)

        return train_x, train_y

    def _update_train_meta(self, train_diff_x, train_diff_y):
        # 更新全量的train 样本
        if self.meta_train_y.shape[0] == 0:
            self.meta_train_x = train_diff_x
            self.meta_train_y = train_diff_y
        else:
            self.meta_train_x = self.meta_train_x + train_diff_x
            self.meta_train_y = np.concatenate([self.meta_train_y, train_diff_y], axis=0)

    def extend_train_data(self, x, y):
        train_x, train_y = map_x_y(self.meta_train_index, x, y)
        if self.do_generate_sample and not self.multi_label:
            train_x, train_y = self.generate_pseudo_samples(train_x, train_y)
            self.do_generate_sample = False
        return train_x, train_y

    def get_sampling_data_frm_full_train(self):
        """
        从全局的train data中采样，只看当前的 meta_train_x, meta_train_y
        :return:
        """
        sample_index = get_sample_index(self.meta_train_y, self.num_classes)
        train_label_distribution = np.sum(np.array(self.meta_train_y), 0)
        self.balance_sampling_index(sample_index, train_label_distribution)
        # 每次只看当前需要采样的数据是否均衡，是否需要生成伪样本
        self.normal_std, self.empty_class_ = get_imbalance_statistic(train_label_distribution)
        self.check_imbalance_level(train_label_distribution)
        self.new_generate_samples_idx = self.generate_presudo_samples(sample_index)
        self.imbalance_flg = False
        train_x, train_y = self.extend_train_data(x=self.meta_train_x, y=self.meta_train_y)
        train_label_distribution = np.sum(np.array(train_y), 0)
        return train_x, train_y

    def sample_dataset_pipeline(self, use_val=False, update_train=True, data_x=None, data_y=None):
        """
        全局采样pipeline
        :param use_val: 是否采用val数据
        :param update_train: 是否更新全量train
        :param data_x: 采样数据来源x：增量数据或者全量原始数据
        :param data_y: 采样数据来源y：增量数据或者全量原始数据
        :return: 均衡采样后的训练集/评估集，use_val为True时，评估集为空
        """
        val_diff_x, val_diff_y = None, None
        ############################ 采样准备阶段 ###################################
        if update_train:
            # 增量更新（第一次样本即增量）
            self.add_index, self.add_val_index = self.sample_val_index(data_y)

            val_diff_x, val_diff_y = map_x_y(self.add_val_index, data_x, data_y)
            # 此时的训练集没有进行采样
            train_diff_x, train_diff_y = flat_map_x_y(index=self.add_index, x=data_x, y=data_y)

            if use_val:  # 如果采用val，即当前不分train valid，全部数据更新meta_train
                train_diff_x = train_diff_x + val_diff_x
                train_diff_y = np.concatenate([train_diff_y, val_diff_y], axis=0)
                val_diff_x = None
                val_diff_y = None

            self._update_train_meta(train_diff_x, train_diff_y)

        if val_diff_x:
            val_label_distribution = np.sum(np.array(val_diff_y), 0)

        ############################ 进入采样阶段 ###################################
        train_x, train_y = self.get_sampling_data_frm_full_train()
        return train_x, train_y, val_diff_x, val_diff_y

    def sample_dataset_iter(self, add_val_to_train=False, update_train=True, use_full=False):
        """

        :param add_val_to_train: 是否采用val数据
        :param update_train: 是否更新全量train：当没有新增样本时，也不再更新全量train
        :param use_full:使用增量数据 or 使用全量数据
        :return:
        """
        if not use_full:
            # 在新样本上重新划分训练集和评估集
            return self.sample_dataset_pipeline(add_val_to_train, update_train,
                                                data_x=self.update_x,
                                                data_y=self.update_y)
        else:
            # 清空历史记录数据
            self.meta_train_y = np.array([])
            self.meta_train_x = []

            return self.sample_dataset_pipeline(use_val=False, update_train=True,
                                                data_x=self.meta_data_x, data_y=self.meta_data_y)

    def vectorize_data(self, x_train, x_val=None, analyzer='word'):
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=20000, analyzer=analyzer)

        if x_val:
            full_text = x_train + x_val
        else:
            full_text = x_train
        vectorizer.fit(full_text)
        train_vectorized = vectorizer.transform(x_train)
        if x_val:
            val_vectorized = vectorizer.transform(x_val)
            return train_vectorized, val_vectorized, vectorizer
        return train_vectorized, vectorizer

    def sequentialize_data_no_padding(self, train_contents, feature_mode, val_contents=[], tokenizer=None,
                                      max_length=None,
                                      Max_Vocab_Size=None):
        if Max_Vocab_Size is None:
            Vocab_Size = MAX_VOCAB_SIZE
        else:
            Vocab_Size = Max_Vocab_Size
        if tokenizer is None:
            if feature_mode == 0:
                tokenizer = text.Tokenizer(num_words=Vocab_Size,
                                           char_level=True,
                                           oov_token="UNK")
            elif feature_mode == 1:
                tokenizer = text.Tokenizer(num_words=Vocab_Size)

            tokenizer.fit_on_texts(train_contents)

        _max_length = max_length
        word_index = tokenizer.word_index
        num_features = min(len(word_index) + 1, Vocab_Size)

        if val_contents:
            return word_index, num_features, tokenizer, _max_length
        else:
            return word_index, num_features, tokenizer, _max_length
