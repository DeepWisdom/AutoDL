from __future__ import print_function
from __future__ import absolute_import

from keras import layers
from keras.regularizers import l2
from keras.layers import Activation, Conv1D, Conv2D, Input, Lambda
from keras.layers import BatchNormalization, Flatten, Dense, Reshape
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D

weight_decay = 1e-3


def identity_block_2D(input_tensor, kernel_size, filters, stage, block, trainable=True):
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    bn_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce/bn'
    x = Conv2D(filters1, (1, 1),
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=trainable,
               kernel_regularizer=l2(weight_decay),
               name=conv_name_1)(input_tensor)
    x = BatchNormalization(axis=bn_axis, trainable=trainable, name=bn_name_1)(x)
    x = Activation('relu')(x)
    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3/bn'
    x = Conv2D(filters2, kernel_size,
               padding='same',
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=trainable,
               kernel_regularizer=l2(weight_decay),
               name=conv_name_2)(x)
    x = BatchNormalization(axis=bn_axis, trainable=trainable, name=bn_name_2)(x)
    x = Activation('relu')(x)
    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase/bn'
    x = Conv2D(filters3, (1, 1),
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=trainable,
               kernel_regularizer=l2(weight_decay),
               name=conv_name_3)(x)
    x = BatchNormalization(axis=bn_axis, trainable=trainable, name=bn_name_3)(x)
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block_2D(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    bn_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce/bn'
    x = Conv2D(filters1, (1, 1),
               strides=strides,
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=trainable,
               kernel_regularizer=l2(weight_decay),
               name=conv_name_1)(input_tensor)
    x = BatchNormalization(axis=bn_axis, trainable=trainable, name=bn_name_1)(x)
    x = Activation('relu')(x)
    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3/bn'
    x = Conv2D(filters2, kernel_size, padding='same',
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=trainable,
               kernel_regularizer=l2(weight_decay),
               name=conv_name_2)(x)
    x = BatchNormalization(axis=bn_axis, trainable=trainable, name=bn_name_2)(x)
    x = Activation('relu')(x)
    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase/bn'
    x = Conv2D(filters3, (1, 1),
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=trainable,
               kernel_regularizer=l2(weight_decay),
               name=conv_name_3)(x)
    x = BatchNormalization(axis=bn_axis, trainable=trainable, name=bn_name_3)(x)
    conv_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_proj'
    bn_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_proj/bn'
    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      kernel_initializer='orthogonal',
                      use_bias=False,
                      trainable=trainable,
                      kernel_regularizer=l2(weight_decay),
                      name=conv_name_4)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, trainable=trainable, name=bn_name_4)(shortcut)
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet_2D_v1(input_dim, mode='train'):
    bn_axis = 3
    if mode == 'train':
        inputs = Input(shape=input_dim, name='input')
    else:
        inputs = Input(shape=(input_dim[0], None, input_dim[-1]), name='input')
    x1 = Conv2D(64, (7, 7),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv1_1/3x3_s1')(inputs)
    x1 = BatchNormalization(axis=bn_axis, name='conv1_1/3x3_s1/bn', trainable=True)(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2))(x1)
    x2 = conv_block_2D(x1, 3, [48, 48, 96], stage=2, block='a', strides=(1, 1), trainable=True)
    x2 = identity_block_2D(x2, 3, [48, 48, 96], stage=2, block='b', trainable=True)
    x3 = conv_block_2D(x2, 3, [96, 96, 128], stage=3, block='a', trainable=True)
    x3 = identity_block_2D(x3, 3, [96, 96, 128], stage=3, block='b', trainable=True)
    x3 = identity_block_2D(x3, 3, [96, 96, 128], stage=3, block='c', trainable=True)
    x4 = conv_block_2D(x3, 3, [128, 128, 256], stage=4, block='a', trainable=True)
    x4 = identity_block_2D(x4, 3, [128, 128, 256], stage=4, block='b', trainable=True)
    x4 = identity_block_2D(x4, 3, [128, 128, 256], stage=4, block='c', trainable=True)
    x5 = conv_block_2D(x4, 3, [256, 256, 512], stage=5, block='a', trainable=True)
    x5 = identity_block_2D(x5, 3, [256, 256, 512], stage=5, block='b', trainable=True)
    x5 = identity_block_2D(x5, 3, [256, 256, 512], stage=5, block='c', trainable=True)
    y = MaxPooling2D((3, 1), strides=(2, 1), name='mpool2')(x5)
    return inputs, y


def resnet_2D_v2(input_dim, mode='train'):
    bn_axis = 3
    if mode == 'train':
        inputs = Input(shape=input_dim, name='input')
    else:
        inputs = Input(shape=(input_dim[0], None, input_dim[-1]), name='input')
    x1 = Conv2D(64, (7, 7), strides=(2, 2),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv1_1/3x3_s1')(inputs)
    x1 = BatchNormalization(axis=bn_axis, name='conv1_1/3x3_s1/bn', trainable=True)(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2))(x1)
    x2 = conv_block_2D(x1, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable=True)
    x2 = identity_block_2D(x2, 3, [64, 64, 256], stage=2, block='b', trainable=True)
    x2 = identity_block_2D(x2, 3, [64, 64, 256], stage=2, block='c', trainable=True)
    x3 = conv_block_2D(x2, 3, [128, 128, 512], stage=3, block='a', trainable=True)
    x3 = identity_block_2D(x3, 3, [128, 128, 512], stage=3, block='b', trainable=True)
    x3 = identity_block_2D(x3, 3, [128, 128, 512], stage=3, block='c', trainable=True)
    x4 = conv_block_2D(x3, 3, [256, 256, 1024], stage=4, block='a', strides=(1, 1), trainable=True)
    x4 = identity_block_2D(x4, 3, [256, 256, 1024], stage=4, block='b', trainable=True)
    x4 = identity_block_2D(x4, 3, [256, 256, 1024], stage=4, block='c', trainable=True)
    x5 = conv_block_2D(x4, 3, [512, 512, 2048], stage=5, block='a', trainable=True)
    x5 = identity_block_2D(x5, 3, [512, 512, 2048], stage=5, block='b', trainable=True)
    x5 = identity_block_2D(x5, 3, [512, 512, 2048], stage=5, block='c', trainable=True)
    y = MaxPooling2D((3, 1), strides=(2, 1), name='mpool2')(x5)
    return inputs, y


def choose_net(token='resnet34s'):
    token_backbone = {
        'resnet34s': resnet_2D_v1,
        'default': resnet_2D_v2
    }
    if token in token_backbone:
        return token_backbone[token]
    else:
        return token_backbone['default']