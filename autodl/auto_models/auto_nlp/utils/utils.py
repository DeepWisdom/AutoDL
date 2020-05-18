# -*- coding: utf-8 -*-
# @Date    : 2020/1/17 14:32
# @Desc    :
import os
import re
import string
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
from keras.preprocessing import text
from keras.preprocessing import sequence
from nltk.stem.snowball import EnglishStemmer, SnowballStemmer
stemmer = SnowballStemmer('english')

MAX_SEQ_LENGTH = 601
MAX_CHAR_LENGTH = 96
MAX_EN_CHAR_LENGTH = 35
import multiprocessing
from multiprocessing import Pool
with open(os.path.join(os.path.dirname(__file__), "en_stop_words_nltk.txt"), "r+",encoding='utf-8') as fp:
    nltk_english_stopwords = fp.readlines()
    nltk_english_stopwords = [word.strip() for word in nltk_english_stopwords]

full_stop_words = list(stopwords)+nltk_english_stopwords
NCPU = multiprocessing.cpu_count() - 1


def set_mp(processes=4):
    import multiprocessing as mp

    def init_worker():
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    global pool
    try:
        pool.terminate()
    except BaseException:
        pass

    if processes:
        pool = mp.Pool(processes=processes, initializer=init_worker)
    else:
        pool = None
    return pool


def clean_zh_text(dat, ratio=0.1, is_ratio=False):
    REPLACE_BY_SPACE_RE = re.compile('[“”【】/（）：！～「」、|，；。"/(){}\[\]\|@,\.;]')

    ret = []
    for line in dat:
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        line = line.strip()

        if is_ratio:
            NUM_CHAR = max(int(len(line) * ratio), MAX_CHAR_LENGTH)
        else:
            NUM_CHAR = MAX_CHAR_LENGTH

        if len(line) > NUM_CHAR:
            line = line[0:NUM_CHAR]
        ret.append(line)
    return ret

def clean_en_text(dat, ratio=0.1, is_ratio=True, vocab=None, rmv_stop_words=True):

    trantab = str.maketrans(dict.fromkeys(string.punctuation+"@!#$%^&*()-<>[]<=>;:?.\/+[\\]^_`{|}~\t\n"+'0123456789'))
    ret = []
    for line in dat:
        line = line.strip()
        line = line.translate(trantab)
        line_split = line.split()
        line_split = [word.lower() for word in line_split if (len(word)<MAX_EN_CHAR_LENGTH and len(word)>1)]
        if vocab is not None:
            # print("use tfidf vocab!")
            _line_split = list(set(line_split).intersection(vocab))
            _line_split.sort(key=line_split.index)
            line_split = _line_split


        if rmv_stop_words:

            new_line_split = list(set(line_split).difference(set(full_stop_words)))
            new_line_split.sort(key=line_split.index)

        else:
            new_line_split = line_split

        if is_ratio:
            NUM_WORD = max(int(len(new_line_split) * ratio), MAX_SEQ_LENGTH)
        else:
            NUM_WORD = MAX_SEQ_LENGTH
        # new_line_split = [stemmer.stem(word) for word in new_line_split]
        if len(new_line_split) > NUM_WORD:
            line = " ".join(new_line_split[0:NUM_WORD])
        else:
            line = " ".join(new_line_split)
        ret.append(line)

    return ret

def chunkIt(seq, num):

    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def clean_en_text_parallel(dat, worker_num=NCPU, partition_num=10, vocab=None):
    sub_data_list = chunkIt(dat, num=partition_num)
    p = Pool(processes=worker_num)

    data = p.map(clean_en_text, sub_data_list)
    p.close()

    flat_data = [item for sublist in data for item in sublist]

    return flat_data

def clean_data(data, language, max_length, vocab, rmv_stop_words=True):
    if language=="EN":
        data = clean_en_text(data, vocab=vocab, rmv_stop_words=rmv_stop_words)
    else:
        data = clean_zh_text(data)
    return data

def clean_en_text_single(line, vocab, ratio=0.1, is_ratio=True, rmv_stop_words=True):
    trantab = str.maketrans(
        dict.fromkeys(string.punctuation + "@!#$%^&*()-<>[]<=>;:?.\/+[\\]^_`{|}~\t\n" + '0123456789'))
    line = line.strip()
    line = line.translate(trantab)
    line_split = line.split()
    line_split = [word.lower() for word in line_split if (len(word) < MAX_EN_CHAR_LENGTH and len(word) > 1)]
    if vocab is not None:
        _line_split = list(set(line_split).intersection(vocab))
        _line_split.sort(key=line_split.index)
        line_split = _line_split


    if rmv_stop_words:
        new_line_split = list(set(line_split).difference(set(full_stop_words)))
        new_line_split.sort(key=line_split.index)
    else:
        new_line_split = line_split

    if is_ratio:
        NUM_WORD = max(int(len(new_line_split) * ratio), MAX_SEQ_LENGTH)
    else:
        NUM_WORD = MAX_SEQ_LENGTH
    if len(new_line_split) > NUM_WORD:
        _line = " ".join(new_line_split[0:NUM_WORD])
    else:
        _line = " ".join(new_line_split)
    return _line

def pad_sequence(data_ids, padding_val, max_length):

    x_ids = sequence.pad_sequences(data_ids, maxlen=max_length, padding='post', value=padding_val)
    return x_ids

