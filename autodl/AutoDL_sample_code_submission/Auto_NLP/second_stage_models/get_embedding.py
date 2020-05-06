# -*- coding: utf-8 -*-
import gzip
import os
import time
import numpy as np
import gc

class GET_EMBEDDING:
    stime = time.time()
    embedding_path = '/app/embedding'
    fasttext_embeddings_index = {}

    fasttext_embeddings_index_zh = {}
    fasttext_embeddings_index_en = {}

    f_zh = gzip.open(os.path.join(embedding_path, 'cc.zh.300.vec.gz'),'rb')
    f_en = gzip.open(os.path.join(embedding_path, 'cc.en.300.vec.gz'),'rb')

    for line in f_zh.readlines():
        values = line.strip().split()
        word = values[0].decode('utf8')
        coefs = np.asarray(values[1:], dtype='float32')
        fasttext_embeddings_index_zh[word] = coefs
    embedding_dict_zh = fasttext_embeddings_index_zh
    del f_zh, values, word, coefs
    gc.collect()
    print('read zh embedding time: {}s.'.format(time.time()-stime))

    for line in f_en.readlines():
        values = line.strip().split()
        word = values[0].decode('utf8')
        coefs = np.asarray(values[1:], dtype='float32')
        fasttext_embeddings_index_en[word] = coefs
    embedding_dict_en = fasttext_embeddings_index_en
    print('read en embedding time: {}s.'.format(time.time()-stime))
