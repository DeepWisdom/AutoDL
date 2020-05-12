import re
import string
import numpy as np

import jieba_fast as jieba

from ...auto_nlp.second_stage_models import ac
from ...at_nlp.data_manager.feature_config import MAX_SEQ_LENGTH


def detect_digits(input_str):
    trantab = str.maketrans(dict.fromkeys(string.punctuation + "@!#$%^&*()-<>[]?.\/+_~:"))
    input_str = input_str.strip()
    clean_line = input_str.translate(trantab)
    cnt = 0
    words = clean_line.strip().split()
    for word in words:
        if word.isdigit():
            cnt += 1
    return round(float(cnt) / float(len(words)), 4)


def detect_supper_and_digits(input_str_list):
    trantab = str.maketrans(dict.fromkeys(string.punctuation + "@!#$%^&*()-<>[]?.\/+_~:"))
    upper_cnt, digits_cnt = [], []
    for input_str in input_str_list:
        input_str = input_str.strip()
        clean_line = input_str.translate(trantab)
        cnt = 0
        digit_cnt = 0
        words = clean_line.strip().split()
        for word in words:
            if word.istitle() or word.isupper():
                cnt += 1
            if word.isdigit():
                digit_cnt += 1
        if len(words) > 0:
            upper_cnt.append(round(float(cnt) / float(len(words)), 5))
            digits_cnt.append(round(float(digit_cnt) / float(len(words)), 5))
    return np.average(upper_cnt), np.average(digits_cnt)


def detect_punctuation(input_str_lst):
    trantab = str.maketrans(dict.fromkeys(string.punctuation + "@!#$%^&*()-<>[]?.\/+_~:" + '0123456789'))
    cnt = []
    for input_str in input_str_lst:
        input_str = input_str.strip()
        clean_line = input_str.translate(trantab)
        cnt_original = len(input_str.split())
        cnt_clean = len(clean_line.split())
        if cnt_original == 0:
            cnt.append(0.0)
        else:
            cnt.append(round(float(cnt_original - cnt_clean) / float(cnt_original), 5))
    return np.average(cnt)


def _tokenize_chinese_words(text):
    return ' '.join(jieba.cut(text, cut_all=False))


def clean_en_text(dat, ratio=0.1, is_ratio=True):
    REPLACE_BY_SPACE_RE = re.compile('["/(){}\[\]\|@,;-]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z #+_]')

    ret = []
    for line in dat:
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        # line = BAD_SYMBOLS_RE.sub('', line)
        line = line.strip()
        line_split = line.split()

        if is_ratio:
            NUM_WORD = max(int(len(line_split) * ratio), MAX_SEQ_LENGTH)
        else:
            NUM_WORD = MAX_SEQ_LENGTH

        if len(line_split) > NUM_WORD:
            line = " ".join(line_split[0:NUM_WORD])

        ret.append(line)
    return ret


def clean_zh_text(dat, seq_len):
    dat_array = np.array(dat, dtype='object')
    return ac.clean_text_zh_seg1(dat_array, seq_len)


def apply_ratio(sentence, ratio=0.1, max_seq_length=MAX_SEQ_LENGTH):
    num_words = max(int(len(sentence) * ratio), max_seq_length)
    return sentence[:num_words]


def apply_filter_word_len(sentence, filter_word_len_min=1, filter_word_len_max=100):
    words = [w for w in sentence if (len(w) > filter_word_len_min and len(w) < filter_word_len_max)]
    return words


def apply_cut_sentence(sentence, cut_style=0, cut_length_pre=301, cut_length_post=0):
    if cut_style == 0:
        return sentence[:cut_length_pre]
    elif cut_style == 1:
        if len(sentence) <= cut_length_pre + cut_length_post:
            return sentence
        return sentence[:cut_length_pre] + sentence[-cut_length_post:]


def clean_en_text_custom(dat, custom_config):
    ret = []
    for line in dat:
        REPLACE_BY_SPACE_RE = re.compile('["/(){}\[\]\|@,;-]')
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        line = line.strip()
        line_split = line.split()
        # 先截断再清洗
        if "cut_style" in custom_config:
            line_split = apply_cut_sentence(line_split,
                                            custom_config["cut_style"]["cut_style"],
                                            custom_config["cut_style"]["cut_length_pre"],
                                            custom_config["cut_style"]["cut_length_post"])

        if "filter_word_len" in custom_config:
            line_split = apply_filter_word_len(line_split, custom_config["filter_word_len"]["word_len_min"],
                                               custom_config["filter_word_len"]["word_len_max"])

        if "is_ratio" in custom_config:
            line_split = apply_ratio(line_split, custom_config["is_ratio"]["ratio"])

        line = " ".join(line_split)
        ret.append(line)
    return ret


def clean_zh_text_custom(dat, custom_config):
    ret = []
    for line in dat:
        line = line.strip()
        if "cut_style" in custom_config:
            line = apply_cut_sentence(line,
                                      custom_config["cut_style"]["cut_style"],
                                      custom_config["cut_style"]["cut_length_pre"],
                                      custom_config["cut_style"]["cut_length_post"])
        ret.append(line)
    ret = clean_zh_text(ret, seq_len=MAX_SEQ_LENGTH)
    return ret
