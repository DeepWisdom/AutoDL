# -*- coding: utf-8 -*-
# @Date    : 2020/1/17 14:19
# @Author  :
# @Desc    :
import numpy as np
import keras
from keras.preprocessing import sequence


class DataGenerator(keras.utils.Sequence):
    def __init__(self, x_train, labels, batch_size,  language, max_length, vocab, num_features,
                 tokenizer=None,
                 shuffle=True):
        self.indices_ = None
        self.batch_size = batch_size
        self.X = x_train
        self.labels = labels
        self.tokenizer = tokenizer
        self.language = language
        self.shuffle = shuffle
        self.max_length = max_length
        self.vocab = vocab
        self.num_features = num_features
        self.on_epoch_end()

    def __len__(self):
        if self.shuffle:
            if len(self.X)>self.batch_size: # 保证至少有一个batch
                return int(np.floor(len(self.X) / self.batch_size))
            else:
                return 1
        else:
            if len(self.X) % self.batch_size==0:
                return int((len(self.X) / self.batch_size))
            else:
                return int((len(self.X) / self.batch_size)+1)

    def __getitem__(self, index):
        indexes = self.indices_[index * self.batch_size:min((index + 1) * self.batch_size, len(self.X))]
        X_temp = [self.X[k] for k in indexes]
        batch_x, batch_y = self._process(X_temp, indexes)
        return batch_x, batch_y

    def on_epoch_end(self):
        self.indices_ = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indices_)

    def _process(self, X_temp, indexes):
        data_ids = self.tokenizer.texts_to_sequences(X_temp)
        max_length = self.max_length
        batch_x = sequence.pad_sequences(data_ids, maxlen=max_length, padding='post')
        batch_y = self.labels[indexes]
        return batch_x, batch_y
