# -*- coding: utf-8 -*-
# @Date    : 2020/3/3 11:23
import keras
from keras.layers import Input, Dense
from keras.layers import Embedding, Flatten, Conv1D, concatenate
from keras.layers import MaxPooling1D, GlobalMaxPooling1D
from keras.layers import Dropout, BatchNormalization
from keras.layers import PReLU

from at_nlp.model_lib.model_utils import _get_last_layer_units_and_activation


def TextCNN_Model(input_shape,
                  embedding_matrix,
                  max_length,
                  num_features,
                  num_classes,
                  input_tensor=None,
                  filter_num=64,
                  emb_size=300,
                  trainable=False,
                  use_multi_label = False
                  ):
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes, use_softmax=True)

    in_text = Input(name='inputs', shape=[max_length], tensor=input_tensor)

    if embedding_matrix is None:
        layer = Embedding(input_dim=num_features,
                          output_dim=emb_size,
                          input_length=input_shape)(in_text)
    else:
        layer = Embedding(input_dim=num_features,
                          output_dim=emb_size,
                          input_length=input_shape,
                          weights=[embedding_matrix],
                          trainable=trainable)(in_text)

    cnns = []
    filter_sizes = [2, 3, 4, 5]
    for size in filter_sizes:
        cnn_l = Conv1D(filter_num,
                       size,
                       padding='same',
                       strides=1,
                       activation='relu')(layer)

        pooling_0 = MaxPooling1D(max_length - size + 1)(cnn_l)
        pooling_0 = Flatten()(pooling_0)
        cnns.append(pooling_0)

    cnn_merge = concatenate(cnns, axis=-1)
    out = Dropout(0.2)(cnn_merge)
    if use_multi_label:
        main_output = Dense(op_units, activation='sigmoid')(out)
    else:
        main_output = Dense(op_units, activation=op_activation)(out)
    md = keras.models.Model(inputs=in_text, outputs=main_output)
    return md


def CNN_Model(max_length, num_classes, num_features, embedding_matrix=None,
              trainable=False, input_shape=None,
              input_tensor=None,
              filter_num=64,
              emb_size=300):
    in_text = Input(shape=(max_length,))
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes, use_softmax=True)

    trainable = True
    if embedding_matrix is None:
        x = Embedding(num_features, 64, trainable=trainable)(in_text)
    else:
        x = Embedding(num_features, 300, trainable=trainable, weights=[embedding_matrix])(in_text)

    x = Conv1D(128, kernel_size=5, padding='valid', kernel_initializer='glorot_uniform')(x)
    x = GlobalMaxPooling1D()(x)

    x = Dense(128)(x)
    x = PReLU()(x)
    x = Dropout(0.35)(x)
    x = BatchNormalization()(x)

    y = Dense(op_units, activation=op_activation)(x)

    md = keras.models.Model(inputs=[in_text], outputs=y)

    return md
