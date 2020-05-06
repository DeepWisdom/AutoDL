# -*- coding: utf-8 -*-
# @Date    : 2020/3/2 11:59
# @Author  : stella
# @Desc    :
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from keras.preprocessing import text
from keras.preprocessing import sequence

from at_nlp.data_manager.preprocess_utils import _tokenize_chinese_words, clean_en_text_custom, clean_zh_text_custom


def preprocess_data(x, preprocess_conf, language='EN', do_seg=False):
    """
    文本清洗
    :param x:list<string>：输入的原始文本
    :param preprocess_conf: :定义的清洗选项及操作
    :param language: 文本对应语言
    :param do_seg: 是否进行分词(针对ZH)
    :return: 清洗后的文本列表
    """
    if language == 'ZH':
        if not do_seg:
            clean_dat = clean_zh_text_custom(x, preprocess_conf)
        else:

            clean_dat = clean_zh_text_custom(x, preprocess_conf)
            clean_dat = list(map(_tokenize_chinese_words, clean_dat))
    else:
        clean_dat = clean_en_text_custom(x, preprocess_conf)

    return clean_dat


def build_tokenizer(dat, tokenizer_type="svm", **kwargs):
    if tokenizer_type == "svm":
        if kwargs["tfidf"]:
            tokenizer = TfidfVectorizer(ngram_range=(1, 1),
                                        max_features=kwargs["max_features"],
                                        analyzer=kwargs["analyzer"])
            tokenizer.fit(dat)
            return tokenizer
        elif kwargs["hashing"]:
            tokenizer = HashingVectorizer(ngram_range=(1, 1),
                                          n_features=kwargs["max_features"],
                                          analyzer=kwargs["analyzer"])
            return tokenizer

    elif tokenizer_type == "nn":
        tokenizer = text.Tokenizer(num_words=kwargs["num_words"])
        tokenizer.fit_on_texts(dat)
        return tokenizer


def postprocess_data(clean_dat, tokenizer, tokenizer_type="svm", **kwargs):
    if tokenizer_type == "svm":
        vectorized_dat = tokenizer.transform(clean_dat)
        return vectorized_dat
    elif tokenizer_type == "nn":
        id_dat = tokenizer.texts_to_sequences(clean_dat)
        max_length = kwargs["pad_max_length"]
        padding_method = kwargs["padding"]
        sequentialize_dat = sequence.pad_sequences(id_dat, maxlen=max_length, padding=padding_method)
        return sequentialize_dat


