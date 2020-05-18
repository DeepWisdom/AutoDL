"""Combine all winner solutions in previous challenges (AutoCV, AutoCV2,
AutoNLP and AutoSpeech).
"""

import numpy as np
import os
import sys
import tensorflow as tf
from scipy import stats
import multiprocessing

from .nlp_autodl_config import DM_DS_PARAS
from .at_nlp.run_model import RunModel as AutoNLPModel
from .nlp_dataset_convertor import TfDatasetsConvertor as TfDatasetTransformer


NCPU = multiprocessing.cpu_count() - 1

feature_dict = {'avg_upper_cnt': 0.13243329407894736, 'check_len': 224.19571428571427, 'imbalance_level': 0,
                'avg_punct_cnt': 0.009241672368421052, 'is_shuffle': False, 'train_num': 40000, 'language': 'EN',
                'kurtosis': -2.0, 'avg_digit_cnt': 0.0056662657894736845, 'seq_len_std': 170,
                'first_detect_normal_std': 0.00025, 'test_num': 10000, 'max_length': 1830, 'avg_length': 230,
                'class_num': 2, 'min_length': 1}

# model_dirs = ['',  # current directory
#               'AutoCV/{}'.format(META_SOLUS.cv_solution),  # AutoCV/AutoCV2 winner model
#               'AutoNLP/{}'.format(META_SOLUS.nlp_solution),  # AutoNLP 2nd place winner
#               # 'AutoSpeech/PASA_NJU',    # AutoSpeech winner
#               'AutoSpeech/{}'.format(META_SOLUS.speech_solution),  # AutoSpeech winner
#               'tabular_Meysam']  # simple NN model
# for model_dir in model_dirs:
#     sys.path.append(os.path.join(here, model_dir))

seq_len = []


def meta_domain_2_model(domain):
    return AutoNLPModel

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  #
config.log_device_placement = False  #
config.gpu_options.per_process_gpu_memory_fraction = 0.9

FIRST_SNOOP_DATA_NUM = 700

INDEX_TO_TOKENS = []
NLP_SEP = " "


def linear_sampling_func(call_num):
    return min(max(call_num * 0.08, 0), 1)


class Model():
    """A model that combine all winner solutions. Using domain inferring and
    apply winner solution in the corresponding domain."""

    def __init__(self, metadata):
        """
        Args:
          metadata: an AutoDLMetadata object. Its definition can be found in
              AutoDL_ingestion_program/dataset.py
        """
        self.done_training = False
        self.metadata = metadata
        self.first_round_sample_maxnum = 200
        self.call_num = -1  # 0
        self.domain_dataset_train_dict = {"x": [],
                                          "y": np.array([])}
        self.domain = "text"

        DomainModel = meta_domain_2_model(self.domain)

        self.domain_metadata = get_domain_metadata(metadata, self.domain)
        self.class_num = self.domain_metadata["class_num"]
        self.train_num = self.domain_metadata["train_num"]

        self.domain_model = DomainModel(self.domain_metadata)
        self.nlp_index_to_token = None
        self.nlp_sep = None
        self.init_nlp()
        self.domain_model.vocab = self.vocabulary
        self.shuffle = False
        self.check_len = 0
        self.imbalance_level = -1

        self.tf_dataset_trainsformer = TfDatasetTransformer(if_train_shuffle=False, config=config)
        self.tf_dataset_trainsformer.init_nlp_data(self.nlp_index_to_token, self.nlp_sep)
        self.time_record = {}
        self.seq_len = []
        self.first_round_X = []
        self.first_round_Y = np.array([])
        self.X_test_raw = None

    def init_nlp(self):

        self.vocabulary = self.metadata.get_channel_to_index_map()
        self.index_to_token = [None] * len(self.vocabulary)

        for token in self.vocabulary:
            index = self.vocabulary[token]
            self.index_to_token[index] = token


        if self.domain_metadata["language"] == "ZH":
            self.nlp_sep = ""

        else:
            self.nlp_sep = " "

    def fit(self, dataset, remaining_time_budget=None):
        """Train method of domain-specific model."""
        # Convert training dataset to necessary format and
        # store as self.domain_dataset_train

        self.tf_dataset_trainsformer.init_train_tfds(dataset, self.train_num)

        if "train_num" not in self.domain_model.feature_dict:
            self.domain_model.feature_dict["train_num"] = self.train_num
            self.domain_model.feature_dict["class_num"] = self.class_num
            self.domain_model.feature_dict["language"] = self.domain_metadata['language']

        self.set_domain_dataset(dataset, is_training=True)


        if self.call_num == -1:
            self.domain_model.train(self.domain_dataset_train_dict["x"], self.domain_dataset_train_dict["y"],
                                    remaining_time_budget=remaining_time_budget)
        else:

            self.domain_model.train(self.domain_dataset_train_dict["x"], self.domain_dataset_train_dict["y"],
                                    remaining_time_budget=remaining_time_budget)
            self.call_num += 1


        self.done_training = self.domain_model.done_training

    def predict(self, dataset, remaining_time_budget=None):
        """Test method of domain-specific model."""
        # Convert test dataset to necessary format and

        self.tf_dataset_trainsformer.init_test_tfds(dataset)

        self.set_domain_dataset(dataset, is_training=False)

        if self.domain in ['text', 'speech'] and \
                (not self.domain_metadata['test_num'] >= 0):
            self.domain_metadata['test_num'] = len(self.X_test)

        if self.call_num == -1:
            Y_pred = self.domain_model.test(self.domain_dataset_test,
                                            remaining_time_budget=remaining_time_budget)
            self.call_num += 1
        else:
            Y_pred = self.domain_model.test(self.domain_dataset_test,
                                            remaining_time_budget=remaining_time_budget)
        if "test_num" not in self.domain_model.feature_dict:
            self.domain_model.feature_dict["test_num"] = self.domain_metadata['test_num']


        self.done_training = self.domain_model.done_training

        return Y_pred

    def check_label_coverage(self, input_y):
        _label_distribution = np.sum(np.array(input_y), 0)
        empty_class_ = [i for i in range(_label_distribution.shape[0]) if _label_distribution[i] == 0]
        normal_std = np.std(_label_distribution) / np.sum(_label_distribution)
        if empty_class_:
            label_coverage = 1-float(len(empty_class_)) / float(self.class_num)
        else:
            label_coverage = 1.0
        return label_coverage, normal_std

    def check_label_distribution(self, input_y):
        _label_distribution = np.sum(np.array(input_y), 0)
        empty_class_ = [i for i in range(_label_distribution.shape[0]) if _label_distribution[i] == 0]  # 包含样本量为0的类别
        self.kurtosis = stats.kurtosis(_label_distribution)
        self.normal_std = np.std(_label_distribution) / np.sum(_label_distribution)
        if len(empty_class_) == 0:  #
            self.shuffle = False
        else:
            self.shuffle = True
        if self.normal_std > 0.3:
            self.imbalance_level = 2  #
        elif self.normal_std > 0.07:
            self.imbalance_level = 1
        else:
            self.imbalance_level = 0


    def check_input_length(self, input_x):
        check_seq_len = []
        for x in input_x:
            x = x[x != -1]
            check_seq_len.append(x.shape[0])
        self.check_len = np.average(check_seq_len)
        self.check_len_std = np.std(check_seq_len)


    def decide_first_num(self):
        snoop_data_num = min(0.01 * self.train_num, FIRST_SNOOP_DATA_NUM)  #
        snoop_X, snoop_Y = self.tf_dataset_trainsformer.get_nlp_train_dataset(snoop_data_num)
        label_coverage, normal_std = self.check_label_coverage(snoop_Y)
        self.check_input_length(snoop_X[:FIRST_SNOOP_DATA_NUM])
        if normal_std>0.3:
            dataset_read_num = min(5000, int(0.1 * self.train_num))
        else:
            if self.class_num == 2 and self.train_num <= 50000:
                if label_coverage == 1.0:  #
                    dataset_read_num = max(int(0.01 * self.train_num), 500)

                    if self.train_num <= 10000:
                        dataset_read_num = min(5000, self.domain_metadata["class_num"] * 3000)
                else:
                    dataset_read_num = min(5000, int(0.1 * self.train_num))

            elif self.class_num == 2 and self.train_num > 50000:
                if label_coverage == 1.0:  #

                    dataset_read_num = min(int(0.01 * self.train_num), 1000)
                else:  #
                    dataset_read_num = min(5000, int(0.1 * self.train_num))


            elif self.class_num > 2 and self.train_num <= 50000:
                if label_coverage == 1.0:  #
                    dataset_read_num = min(int((2 / self.class_num) * self.train_num), 1000)
                    #
                    if self.train_num <= 10000:
                        dataset_read_num = min(5000, self.domain_metadata["class_num"] * 3000)
                else:
                    dataset_read_num = min(5000, int(0.1 * self.train_num))
            elif self.class_num > 2 and self.train_num > 50000:
                if label_coverage == 1.0:  #
                    #
                    dataset_read_num = min(int((2 / self.class_num) * self.train_num), 1500)
                else:  #
                    dataset_read_num = min(5000, int(0.1 * self.train_num))

                if self.domain_metadata["language"] == "ZH" and self.check_len<=40:
                    dataset_read_num += min(2000, 0.1*self.train_num)

        X, Y = self.tf_dataset_trainsformer.get_nlp_train_dataset(dataset_read_num)
        X = X + snoop_X
        Y = np.concatenate([Y, snoop_Y], axis=0)
        return dataset_read_num, X, Y

    def set_domain_dataset(self, dataset, is_training=True):
        """Recover the dataset in corresponding competition format (esp. AutoNLP
        and AutoSpeech) and set corresponding attributes:
          self.domain_dataset_train
          self.domain_dataset_test
        according to `is_training`.
        """
        # self.dataset = None
        if is_training:
            subset = 'train'
        else:
            subset = 'test'
        attr_dataset = 'domain_dataset_{}'.format(subset)

        if not hasattr(self, attr_dataset):
            if self.domain == 'text':
                if DM_DS_PARAS.text.if_sample and is_training:

                    dataset_read_num, X, Y = self.decide_first_num()


                    self.check_label_distribution(np.array(Y))
                    self.domain_model.imbalance_level = self.imbalance_level

                    feature_dict["check_len"] = float(self.check_len)
                    feature_dict["kurtosis"] = float(self.kurtosis)
                    feature_dict["first_detect_normal_std"] = float(self.normal_std)
                    feature_dict["imbalance_level"] = self.imbalance_level
                    feature_dict["is_shuffle"] = self.shuffle

                    if self.shuffle and self.domain_metadata["language"] == "ZH":
                        self.shuffle = False
                        dataset_read_num = int(0.4 * self.train_num)

                        _X, _Y = self.tf_dataset_trainsformer.get_nlp_train_dataset(dataset_read_num)
                        X = X + _X
                        Y = np.concatenate([Y, _Y], axis=0)


                        _label_distribution = np.sum(Y, 0)
                        occu_class_ = [i for i in range(_label_distribution.shape[0]) if _label_distribution[i] != 0] #

                        if len(occu_class_)>=2:
                            pass
                        else:
                            #
                            dataset_read_num = int(0.2 * self.train_num)
                            _X, _Y = self.tf_dataset_trainsformer.get_nlp_train_dataset(dataset_read_num)
                            X = X + _X
                            Y = np.concatenate([Y, _Y], axis=0)
                            _label_distribution = np.sum(Y, 0)
                            occu_class_ = [i for i in range(_label_distribution.shape[0]) if
                                           _label_distribution[i] != 0]
                            if len(occu_class_)<2:
                                dataset_read_num = int(self.train_num)
                                _X, _Y = self.tf_dataset_trainsformer.get_nlp_train_dataset(dataset_read_num)
                                X = X + _X
                                Y = np.concatenate([Y, _Y], axis=0)


                    if self.shuffle:

                        del self.tf_dataset_trainsformer
                        self.tf_dataset_trainsformer = TfDatasetTransformer(if_train_shuffle=True, config=config)

                        shuffle_size = max(int(0.5 * (self.train_num)), 10000)

                        shuffle_dataset = dataset.shuffle(shuffle_size)


                        self.tf_dataset_trainsformer.init_train_tfds(shuffle_dataset, self.train_num, pad_num=20)

                        X, Y = self.tf_dataset_trainsformer.get_nlp_train_dataset(dataset_read_num)

                        _label_distribution = np.sum(Y, 0)
                        occu_class_ = [i for i in range(_label_distribution.shape[0]) if
                                       _label_distribution[i] != 0]  #
                        if len(occu_class_) >= 2:
                            pass
                        else:
                            dataset_read_num = int(1 * (self.train_num))
                            _X, _Y = self.tf_dataset_trainsformer.get_nlp_train_dataset(dataset_read_num)
                            X = X + _X
                            Y = np.concatenate([Y, _Y], axis=0)


                    self.domain_model.avg_word_per_sample = float(
                        len(self.vocabulary) / self.domain_metadata["train_num"])
                    if "avg_word_per_sample" not in feature_dict:
                        feature_dict["avg_word_per_sample"] = self.domain_model.avg_word_per_sample
                    self.domain_model.feature_dict = feature_dict

                elif not is_training:
                    pad_num = 20
                    X, Y = self.tf_dataset_trainsformer.get_nlp_test_dataset(pad_num=pad_num)

                if is_training:
                    self.first_round_X = X
                    self.first_round_Y = Y
                #

                if self.call_num == 0:
                    corpus = []
                    seq_len = []

                    for _x in X:
                        _x = _x[_x != -1]
                        num_words = max(int(_x.shape[0] * 0.1), 301)
                        _x = _x[:num_words]
                        _x = _x.astype(str)
                        tokens = _x.tolist()
                        document = self.nlp_sep.join(tokens)
                        corpus.append(document)

                else:
                    corpus, seq_len = to_corpus(X, self.index_to_token, self.nlp_sep)


                self.seq_len = seq_len


                if is_training:
                    labels = np.array(Y)
                    cnt = np.sum(np.count_nonzero(labels, axis=1), axis=0)
                    if cnt > labels.shape[0]:
                        self.domain_model.multi_label = True

                        self.domain_model.second_stage_model = None
                        self.domain_model.ft_model = None
                    domain_dataset = corpus, labels

                    self.domain_dataset_train_dict["x"] = corpus
                    self.domain_dataset_train_dict["y"] = labels
                else:
                    domain_dataset = corpus

                    self.domain_dataset_train_dict["x"] = corpus
                    self.X_test = corpus

                setattr(self, attr_dataset, domain_dataset)

            elif self.domain == 'speech':

                setattr(self, attr_dataset, dataset)

            elif self.domain in ['image', 'video', 'tabular']:
                setattr(self, attr_dataset, dataset)
            else:
                raise ValueError("The domain {} doesn't exist.".format(self.domain))

        else:
            if subset == 'test':
                if self.X_test_raw:
                    self.domain_dataset_test, test_seq_len = to_corpus(self.X_test_raw, self.index_to_token,
                                                                       self.nlp_sep)
                self.X_test_raw = None
                return

            if self.domain == 'text':
                if DM_DS_PARAS.text.if_sample and is_training:
                    if self.domain_model.multi_label:
                        self.domain_model.use_multi_svm = True
                        self.domain_model.start_cnn_call_num = 2
                        dataset_read_num = self.train_num
                        if dataset_read_num>50000:
                            dataset_read_num = 50000
                    else:
                        if self.imbalance_level >= 1:
                            dataset_read_num = self.train_num
                            self.domain_model.use_multi_svm = False
                            self.domain_model.start_cnn_call_num = 1
                            if dataset_read_num > 50000:
                                dataset_read_num = 50000
                        else:
                            self.domain_model.use_multi_svm = True
                            if self.call_num <= self.domain_model.start_first_stage_call_num - 1:
                                dataset_read_num = 3000
                                if self.check_len <= 40 or self.normal_std >= 0.2:
                                    dataset_read_num += min(int(0.2 * self.train_num), 12000)
                            else:
                                if self.call_num == self.domain_model.start_first_stage_call_num:
                                    dataset_read_num = int(0.9 * self.domain_metadata["train_num"])
                                    if dataset_read_num > 50000:
                                        dataset_read_num = 50000
                                else:
                                    if self.train_num <= 55555:
                                        dataset_read_num = 4000
                                    else:
                                        dataset_read_num = 5500


                    X, Y = self.tf_dataset_trainsformer.get_nlp_train_dataset(dataset_read_num)


                    if self.call_num == 1:
                        pass


                corpus, seq_len = to_corpus(X, self.index_to_token, self.nlp_sep)
                self.seq_len.extend(seq_len)

                if "avg_length" not in self.domain_model.feature_dict:
                    self.domain_model.feature_dict["avg_length"] = int(np.average(self.seq_len))
                    self.domain_model.feature_dict["max_length"] = int(np.max(self.seq_len))
                    self.domain_model.feature_dict["min_length"] = int(np.min(self.seq_len))
                    self.domain_model.feature_dict["seq_len_std"] = int(np.std(self.seq_len))

                if self.domain_model.max_length == 0:
                    if int(np.max(self.seq_len)) <= 301:
                        self.domain_model.max_length = int(np.max(self.seq_len))
                        self.domain_model.ft_model_check_length = int(np.max(self.seq_len))

                    else:
                        self.domain_model.max_length = int(np.average(self.seq_len))
                        self.domain_model.ft_model_check_length = int(np.average(self.seq_len))

                    self.domain_model.seq_len_std = int(np.std(self.seq_len))


                if is_training:
                    labels = np.array(Y)
                    domain_dataset = corpus, labels
                    self.domain_dataset_train_dict["x"] = corpus
                    self.domain_dataset_train_dict["y"] = labels
                else:
                    domain_dataset = corpus


def to_corpus(x, index_to_token=INDEX_TO_TOKENS, nlp_sep=NLP_SEP):
    corpus = []
    seq_len = []
    for _x in x:
        _x = _x[_x != -1]
        tokens = [index_to_token[i] for i in _x]
        seq_len.append(len(tokens))
        document = nlp_sep.join(tokens)
        corpus.append(document)
    return corpus, seq_len


def infer_domain(metadata):
    """Infer the domain from the shape of the 4-D tensor.

    Args:
      metadata: an AutoDLMetadata object.
    """
    row_count, col_count = metadata.get_matrix_size(0)
    sequence_size = metadata.get_sequence_size()
    channel_to_index_map = metadata.get_channel_to_index_map()
    domain = None
    if sequence_size == 1:
        if row_count == 1 or col_count == 1:
            domain = "tabular"
        else:
            domain = "image"
    else:
        if row_count == 1 and col_count == 1:
            if len(channel_to_index_map) > 0:
                domain = "text"
            else:
                domain = "speech"
        else:
            domain = "video"
    return domain


def is_chinese(tokens):
    """Judge if the tokens are in Chinese. The current criterion is if each token
    contains one single character, because when the documents are in Chinese,
    we tokenize each character when formatting the dataset.
    """
    is_of_len_1 = all([len(t) == 1 for t in tokens[:100]])
    num = [1 for t in tokens[:100] if len(t) == 1]
    ratio = np.sum(num) / 100
    if ratio > 0.95:
        return True
    else:
        return False



def get_domain_metadata(metadata, domain, is_training=True):
    """Recover the metadata in corresponding competitions, esp. AutoNLP
    and AutoSpeech.

    Args:
      metadata: an AutoDLMetadata object.
      domain: str, can be one of 'image', 'video', 'text', 'speech' or 'tabular'.
    """
    if domain == 'text':
        # Fetch metadata info from `metadata`
        class_num = metadata.get_output_size()
        num_examples = metadata.size()
        channel_to_index_map = metadata.get_channel_to_index_map()
        revers_map = {v: k for k, v in channel_to_index_map.items()}
        tokens = [revers_map[int(id)] for id in range(100)]
        language = 'ZH' if is_chinese(tokens) else 'EN'
        time_budget = 1200  # WARNING: Hard-coded

        # Create domain metadata
        domain_metadata = {}
        domain_metadata['class_num'] = class_num
        if is_training:
            domain_metadata['train_num'] = num_examples
            domain_metadata['test_num'] = -1
        else:
            domain_metadata['train_num'] = -1
            domain_metadata['test_num'] = num_examples
        domain_metadata['language'] = language
        domain_metadata['time_budget'] = time_budget

        return domain_metadata

    else:
        return metadata
