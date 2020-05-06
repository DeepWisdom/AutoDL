# -*- coding: utf-8 -*-
MAX_SEQ_LENGTH = 301
MAX_SEQ_EN_LENGTH_PRE = 128
MAX_SEQ_EN_LENGTH_POST = 256
MAX_SEQ_ZH_LENGTH_PRE = 128
MAX_SEQ_ZH_LENGTH_POST = 128

DEFAULT_EN_CONF = {
    'is_ratio':
        {'ratio': 0.1},
}

CUSTOM_EN_CONF = {
    'is_ratio':
        {'ratio': 0.1},
    'filter_word_len':
        {'word_len_min': 1,
         'word_len_max': 300},

    'cut_style':
        {'cut_style': 1,
         'cut_length_pre': MAX_SEQ_EN_LENGTH_PRE,
         'cut_length_post': MAX_SEQ_EN_LENGTH_POST}}

DEFAULT_ZH_CONF = {'cut_style': {'cut_style': 0,
                                   'cut_length_pre': MAX_SEQ_LENGTH,
                                   'cut_length_post': 0}}

CUSTOM_ZH_CONF = {'cut_style': {'cut_style': 1,
                                  'cut_length_pre': MAX_SEQ_ZH_LENGTH_PRE,
                                  'cut_length_post': MAX_SEQ_ZH_LENGTH_POST}}
