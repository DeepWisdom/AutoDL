# -*- coding: utf-8 -*-
# @Date    : 2020/3/3 11:32
import keras
from keras.layers import Input, Dense
from keras.layers import Embedding, Flatten,  Conv2D
from keras.layers import GlobalMaxPooling1D, MaxPooling2D
from keras.layers import Dropout, BatchNormalization, Concatenate, Reshape
from keras.layers import PReLU
from keras.layers import CuDNNGRU, Bidirectional
from keras import backend as K

from ...at_nlp.model_lib.model_utils import _get_last_layer_units_and_activation


def TextRCNN_Model(input_shape,
                   embedding_matrix,
                   max_length,
                   num_features,
                   num_classes,
                   input_tensor=None,
                   emb_size=300,
                   filter_num=64,
                   rnn_units=128,
                   trainable=False):
    inputs = Input(name='inputs', shape=[max_length], tensor=input_tensor)
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes, only_use_softmax=True)

    if embedding_matrix is None:
        layer = Embedding(input_dim=num_features,
                          output_dim=emb_size,
                          input_length=input_shape)(inputs)
    else:

        layer = Embedding(input_dim=num_features,
                          output_dim=emb_size,
                          weights=[embedding_matrix],
                          trainable=trainable)(inputs)

    layer_cell = CuDNNGRU
    embedding_output = layer
    # 拼接
    x_feb = Bidirectional(layer_cell(units=rnn_units,
                                     return_sequences=True))(embedding_output)

    x_feb = Concatenate(axis=2)([x_feb, embedding_output])

    ####使用多个卷积核##################################################
    x_feb = Dropout(rate=0.5)(x_feb)

    dim_2 = K.int_shape(x_feb)[2]

    len_max = max_length
    x_feb_reshape = Reshape((len_max, dim_2, 1))(x_feb)
    # 提取n-gram特征和最大池化， 一般不用平均池化
    conv_pools = []
    filters = [2, 3, 4, 5]

    for filter_size in filters:
        conv = Conv2D(filters=filter_num,
                      kernel_size=(filter_size, dim_2),
                      padding='valid',
                      kernel_initializer='normal',
                      activation='relu',
                      )(x_feb_reshape)

        print("check conv", conv.get_shape())
        pooled = MaxPooling2D(pool_size=(len_max - filter_size + 1, 1),
                              strides=(1, 1),
                              padding='valid',
                              )(conv)
        print("check pooled", pooled.get_shape())
        conv_pools.append(pooled)

    # 拼接
    x = Concatenate()(conv_pools)
    x = Flatten()(x)
    #########################################################################
    output = Dense(op_units, activation=op_activation)(x)
    md = keras.models.Model(inputs=inputs, outputs=output)
    return md


def RNN_Model(max_length, num_classes, num_features, embedding_matrix=None,
              trainable=False, input_shape=None,
              input_tensor=None,
              filter_num=64,
              emb_size=300,
              only_use_softmax=True
              ):
    in_text = Input(shape=(max_length,))
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes, only_use_softmax=only_use_softmax)

    trainable = True
    if embedding_matrix is None:
        x = Embedding(num_features, 64, trainable=trainable)(in_text)
    else:
        x = Embedding(num_features, 300, trainable=trainable, weights=[embedding_matrix])(in_text)

    x = CuDNNGRU(128, return_sequences=True)(x)
    x = GlobalMaxPooling1D()(x)

    x = Dense(128)(x)  #
    x = PReLU()(x)
    x = Dropout(0.35)(x)  # 0
    x = BatchNormalization()(x)

    y = Dense(op_units, activation=op_activation)(x)

    md = keras.models.Model(inputs=[in_text], outputs=y)

    return md
