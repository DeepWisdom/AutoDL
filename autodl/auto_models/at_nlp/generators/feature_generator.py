# -*- coding: utf-8 -*-
# @Date    : 2020/3/3 16:29
from autodl.utils.log_utils import info
from ...at_nlp.data_manager.feature_utils import *
from ...at_nlp.data_manager.preprocess_utils import *
from ...at_nlp.data_manager.feature_config import DEFAULT_EN_CONF, DEFAULT_ZH_CONF, CUSTOM_ZH_CONF

MAX_VOCAB_SIZE = 20000
MAX_SEQ_LENGTH = 301
MAX_TOLERANT_STD = 150


class FeatureGenerator():
    def __init__(self, language, do_seg, num_class):
        self.language = language
        self.num_classes = num_class
        if self.language == "EN":
            self.default_preprocess_conf = DEFAULT_EN_CONF
        else:
            self.default_preprocess_conf = CUSTOM_ZH_CONF
        self.do_seg = do_seg
        self.tokenizer = None
        self.tokenizer_type = ""
        self.tokenizer_conf = {}
        self.data_feature = {}

        self.max_length = MAX_SEQ_LENGTH
        self.seq_len_std = 0.0
        self.vocab_size = MAX_VOCAB_SIZE
        self.num_features = 0
        self.word_index = None

    def reset_tokenizer(self):
        self.tokenizer = None
        self.tokenizer_type = ""
        self.tokenizer_conf = {}
        self.data_feature = {}


    def _set_tokenizer_conf(self, model_name='svm'):
        if model_name == "svm":
            tokenizer_conf = {
                'tfidf': True,
                'max_features': 20000,
                'analyzer': 'word'}
            tokenizer_type = 'svm'

        else:
            tokenizer_conf = {
                'num_words': 20000,
                'use_char': False,
                'pad_max_length': 301,
                'padding': 'post'}
            tokenizer_type = 'nn'

        return tokenizer_type, tokenizer_conf

    def update_preprocess_conf(self):
        pass

    def preprocess_data(self, x):
        return preprocess_data(x, self.default_preprocess_conf, self.language, do_seg=self.do_seg)

    def set_tokenizer(self, dat, tokenizer_type):
        if tokenizer_type == "svm":
            self.tokenizer = build_tokenizer(dat, tokenizer_type, **self.tokenizer_conf)

        elif tokenizer_type == 'nn':
            self.set_max_seq_len()
            self.set_max_vocab_size(dat)
            self.tokenizer_conf['num_words'] = self.vocab_size
            self.tokenizer_conf['pad_max_length'] = self.max_length
            self.tokenizer = build_tokenizer(dat, tokenizer_type, **self.tokenizer_conf)
            self.word_index = self.tokenizer.word_index
            self.num_features = min(len(self.word_index) + 1, self.vocab_size)


    def build_tokenizer(self, clean_dat, model_name, analyzer='word'):
        if self.tokenizer is None:
            self.tokenizer_type, self.tokenizer_conf = self._set_tokenizer_conf(model_name=model_name)
            if model_name == "svm":
                self.tokenizer_conf["analyzer"] = analyzer
            self.set_tokenizer(clean_dat, self.tokenizer_type)

    def postprocess_data(self, clean_dat):
        return postprocess_data(clean_dat, self.tokenizer, tokenizer_type=self.tokenizer_type,  **self.tokenizer_conf)

    def set_data_feature(self):
        self.data_feature["num_features"] = self.num_features
        self.data_feature["num_class"] = self.num_classes
        self.data_feature['max_length'] = self.max_length
        self.data_feature['input_shape'] = self.max_length
        self.data_feature["rnn_units"] = 128
        self.data_feature["filter_num"] = 64
        self.data_feature["word_index"] = self.word_index


    def set_max_seq_len(self):
        if self.max_length > MAX_SEQ_LENGTH:
            self.max_length = MAX_SEQ_LENGTH
        if self.seq_len_std > MAX_TOLERANT_STD:
            self.max_length = MAX_SEQ_LENGTH

    def set_max_vocab_size(self, input_x):
        avg_punct_cnt = detect_punctuation(input_x)
        avg_upper_cnt, avg_digit_cnt = detect_supper_and_digits(input_x)
        info("avg_punct_cnt is {} and avg_upper_cnt is {} and avg_digit_cnt is {}".format(avg_punct_cnt,
                                                                                          avg_upper_cnt,
                                                                                          avg_digit_cnt))
        if avg_punct_cnt <= 0.02:
            Max_Vocab_Size = 30000
        else:
            Max_Vocab_Size = 20000
        self.vocab_size = Max_Vocab_Size
