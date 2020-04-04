# -*- coding: utf-8 -*-
# @Date    : 2020/3/3 11:13
import os
import gzip
import numpy as np

from log_utils import info

EMBEDDING_DIM = 300


def _load_emb(language):
    # loading pretrained embedding

    FT_DIR = '/app/embedding'
    fasttext_embeddings_index = {}
    if language == 'ZH':
        f = gzip.open(os.path.join(FT_DIR, 'cc.zh.300.vec.gz'), 'rb')
    elif language== 'EN':
        f = gzip.open(os.path.join(FT_DIR, 'cc.en.300.vec.gz'), 'rb')
    else:
        raise ValueError('Unexpected embedding path:'
                         ' {unexpected_embedding}. '.format(
            unexpected_embedding=FT_DIR))

    for line in f.readlines():
        values = line.strip().split()
        if language== 'ZH':
            word = values[0].decode('utf8')
        else:
            word = values[0].decode('utf8')
        coefs = np.asarray(values[1:], dtype='float32')
        fasttext_embeddings_index[word] = coefs

    info('Found %s fastText word vectors.' %
         len(fasttext_embeddings_index))
    return fasttext_embeddings_index

def generate_emb_matrix(num_features, word_index, fasttext_embeddings_index):
    cnt = 0
    embedding_matrix = np.zeros((num_features, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= num_features:
            continue
        embedding_vector = fasttext_embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.uniform(
                -0.02, 0.02, size=EMBEDDING_DIM)
            cnt += 1
    print("check self embedding_vector ", embedding_matrix.shape)
    oov_cnt = cnt

    print('fastText oov words: %s' % cnt)
    return oov_cnt, embedding_matrix