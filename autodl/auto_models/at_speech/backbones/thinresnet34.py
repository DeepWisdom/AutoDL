from __future__ import print_function
from __future__ import absolute_import
import keras
import tensorflow as tf
import keras.backend as K

from ...at_speech.backbones import tr34_bb as backbone

weight_decay = 1e-3


class ModelMGPU(keras.Model):
    def __init__(self, ser_model, gpus):
        pmodel = keras.utils.multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)


class VladPooling(keras.engine.Layer):
    def __init__(self, mode, k_centers, g_centers=0, **kwargs):
        self.k_centers = k_centers
        self.g_centers = g_centers
        self.mode = mode
        super(VladPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.cluster = self.add_weight(shape=[self.k_centers+self.g_centers, input_shape[0][-1]],
                                       name='centers',
                                       initializer='orthogonal')
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape
        return (input_shape[0][0], self.k_centers*input_shape[0][-1])

    def call(self, x):
        feat, cluster_score = x
        num_features = feat.shape[-1]
        max_cluster_score = K.max(cluster_score, -1, keepdims=True)
        exp_cluster_score = K.exp(cluster_score - max_cluster_score)
        A = exp_cluster_score / K.sum(exp_cluster_score, axis=-1, keepdims = True)
        A = K.expand_dims(A, -1)
        feat_broadcast = K.expand_dims(feat, -2)
        feat_res = feat_broadcast - self.cluster
        weighted_res = tf.multiply(A, feat_res)
        cluster_res = K.sum(weighted_res, [1, 2])

        if self.mode == 'gvlad':
            cluster_res = cluster_res[:, :self.k_centers, :]

        cluster_l2 = K.l2_normalize(cluster_res, -1)
        outputs = K.reshape(cluster_l2, [-1, int(self.k_centers) * int(num_features)])
        return outputs


def amsoftmax_loss(y_true, y_pred, scale=30, margin=0.35):
    y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
    y_pred *= scale
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)


def vggvox_resnet2d_icassp(input_dim=(257, 250, 1), num_class=8631, mode='train', config=None, net=None):
    if not net:
        net=config['net']
    loss=config['loss']
    vlad_clusters=config['vlad_cluster']
    ghost_clusters=config['ghost_cluster']
    bottleneck_dim=config['bottleneck_dim']
    aggregation = config['aggregation_mode']
    backbone_net = backbone.choose_net(net)
    inputs, x = backbone_net(input_dim=input_dim, mode=mode)
    x_fc = keras.layers.Conv2D(bottleneck_dim, (7, 1),
                               strides=(1, 1),
                               activation='relu',
                               kernel_initializer='orthogonal',
                               use_bias=True, trainable=True,
                               kernel_regularizer=keras.regularizers.l2(weight_decay),
                               bias_regularizer=keras.regularizers.l2(weight_decay),
                               name='x_fc')(x)

    if aggregation == 'avg':
        if mode == 'train':
            x = keras.layers.AveragePooling2D((1, 5), strides=(1, 1), name='avg_pool')(x)
            x = keras.layers.Reshape((-1, bottleneck_dim))(x)
        else:
            x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
            x = keras.layers.Reshape((1, bottleneck_dim))(x)

    elif aggregation == 'vlad':
        x_k_center = keras.layers.Conv2D(vlad_clusters, (7, 1),
                                         strides=(1, 1),
                                         kernel_initializer='orthogonal',
                                         use_bias=True, trainable=True,
                                         kernel_regularizer=keras.regularizers.l2(weight_decay),
                                         bias_regularizer=keras.regularizers.l2(weight_decay),
                                         name='vlad_center_assignment')(x)
        x = VladPooling(k_centers=vlad_clusters, mode='vlad', name='vlad_pool')([x_fc, x_k_center])

    elif aggregation == 'gvlad':
        x_k_center = keras.layers.Conv2D(vlad_clusters+ghost_clusters, (7, 1),
                                         strides=(1, 1),
                                         kernel_initializer='orthogonal',
                                         use_bias=True, trainable=True,
                                         kernel_regularizer=keras.regularizers.l2(weight_decay),
                                         bias_regularizer=keras.regularizers.l2(weight_decay),
                                         name='gvlad_center_assignment')(x)
        x = VladPooling(k_centers=vlad_clusters, g_centers=ghost_clusters, mode='gvlad', name='gvlad_pool')([x_fc, x_k_center])

    else:
        raise IOError('==> unknown aggregation mode')

    x = keras.layers.Dense(bottleneck_dim, activation='relu',
                           kernel_initializer='orthogonal',
                           use_bias=True, trainable=True,
                           kernel_regularizer=keras.regularizers.l2(weight_decay),
                           bias_regularizer=keras.regularizers.l2(weight_decay),
                           name='fc6')(x)

    if loss == 'softmax':
        y = keras.layers.Dense(num_class, activation='softmax',
                               kernel_initializer='orthogonal',
                               use_bias=False, trainable=True,
                               kernel_regularizer=keras.regularizers.l2(weight_decay),
                               bias_regularizer=keras.regularizers.l2(weight_decay),
                               name='prediction')(x)
        trnloss = 'categorical_crossentropy'

    elif loss == 'amsoftmax':
        x_l2 = keras.layers.Lambda(lambda x: K.l2_normalize(x, 1))(x)
        y = keras.layers.Dense(num_class,
                               kernel_initializer='orthogonal',
                               use_bias=False, trainable=True,
                               kernel_constraint=keras.constraints.unit_norm(),
                               kernel_regularizer=keras.regularizers.l2(weight_decay),
                               bias_regularizer=keras.regularizers.l2(weight_decay),
                               name='prediction')(x_l2)
        trnloss = amsoftmax_loss

    else:
        raise IOError('==> unknown loss.')

    if mode == 'pretrain':
        model = keras.models.Model(inputs, x, name='vggvox_resnet2D_{}_{}'.format(loss, aggregation))
    if mode == 'pred':
        model = keras.models.Model(inputs, y, name='vggvox_resnet2D_{}_{}'.format(loss, aggregation))
    return model


def build_tr34_model(net_name, input_dim, num_class, tr34_bb_config):
    return vggvox_resnet2d_icassp(
        input_dim=input_dim,
        num_class=num_class,
        mode="pretrain",
        config=tr34_bb_config,
        net=net_name,
    )
