import librosa
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Activation, Add

from ...auto_tabular.CONSTANT import *
from ...auto_tabular.utils.data_utils import ohe2cat
from .meta_model import MetaModel
from sklearn.metrics import roc_auc_score


class DnnModel(MetaModel):
    def __init__(self):
        super(DnnModel, self).__init__()
        self.max_length = None
        self.mean = None
        self.std = None

        self._model = None
        self.is_init = False

        self.name = 'dnn'
        self.type = 'nn_keras'
        self.patience = 500

        self.max_run = 1000

        self.all_data_round = 950

        self.not_gain_threhlod = 500

        self.is_multi_label = None

    def init_model(self,
                   num_classes,
                   shape=None,
                   is_multi_label=False,
                   **kwargs):
        self.is_multi_label = is_multi_label
        weight_decay = 1e-3
        inputs = Input(shape=(shape,))
        x = BatchNormalization(axis=1)(inputs)
        x = Dropout(0.4)(x)
        x_skip = self.fc(x, 500, weight_decay)
        x = self.fc(x_skip, 500, weight_decay)
        x = Dropout(0.4)(x)
        x = Add()([x, x_skip])
        x_mid = self.fc(x, shape, weight_decay)
        x = self.fc(x_mid, 500, weight_decay)
        for i in range(3):
            x = self.fc(x, 500, weight_decay)

        if not self.is_multi_label:
            logits = Dense(num_classes,
                           #activation="softmax",
                           #kernel_initializer="orthogonal",
                           use_bias=True,
                           trainable=True,
                           kernel_regularizer=keras.regularizers.l2(weight_decay),
                           bias_regularizer=keras.regularizers.l2(weight_decay),
                           name="prediction")(x)
            self._model = tf.keras.Model(inputs=inputs, outputs=logits)
            opt = tf.keras.optimizers.Adam()
            loss = sigmoid_cross_entropy_with_logits
        else:
            preds = Dense(num_classes,
                           activation="softmax",
                           #kernel_initializer="orthogonal",
                           use_bias=True,
                           trainable=True,
                           kernel_regularizer=keras.regularizers.l2(weight_decay),
                           bias_regularizer=keras.regularizers.l2(weight_decay),
                           name="prediction")(x)
            self._model = tf.keras.Model(inputs=inputs, outputs=preds)
            opt = tf.keras.optimizers.Adam()
            loss = tf.keras.losses.CategoricalCrossentropy()

        self._model.compile(optimizer=opt, loss=loss, metrics=["acc"])
        self.is_init = True

    def fc(self, x, out_dim, weight_decay):
        x = Dropout(0.2)(x)
        x = Dense(out_dim,
                  kernel_regularizer=keras.regularizers.l2(weight_decay),
                  use_bias=False,
                  )(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation('relu')(x)
        return x

    def epoch_train(self, dataloader, run_num, **kwargs):
        X, y, train_idxs = dataloader['X'], dataloader['y'], dataloader['train_idxs']
        train_x, train_y = X.loc[train_idxs].values, y[train_idxs]
        print('epoch train shape')
        print(train_x.shape)

        epochs = 5

        if not self.is_multi_label:
            train_y = ohe2cat(train_y)
        self._model.fit(train_x, train_y,
                        epochs=epochs,
                        #callbacks=callbacks,
                        #validation_data=(val_x, ohe2cat(val_y)),
                        # validation_split=0.2,
                        verbose=1,  # Logs once per epoch.
                        batch_size=128,
                        shuffle=True,
                        # initial_epoch=self.epoch_cnt,
                        # use_multiprocessing=True
                        )

    def epoch_valid(self, dataloader):
        X, y, val_idxs= dataloader['X'], dataloader['y'], dataloader['val_idxs']
        val_x, val_y = X.loc[val_idxs].values, y[val_idxs]
        preds = self._model.predict(val_x)
        if self.is_multi_label:
            preds = sigmoid(preds)
        valid_auc = roc_auc_score(val_y, preds)
        return valid_auc

    def predict(self, dataloader, batch_size=32):
        X, test_idxs = dataloader['X'], dataloader['test_idxs']
        test_x = X.loc[test_idxs].values
        preds = self._model.predict(test_x)
        if self.is_multi_label:
            preds = sigmoid(preds)
        return preds


def sigmoid_cross_entropy_with_logits(y_true, y_pred):
    #labels = tf.cast(labels, dtype=tf.float32)
    relu_preds = tf.nn.relu(y_pred)
    exp_logits = tf.exp(- tf.abs(y_pred))
    sigmoid_logits = tf.log(1 + exp_logits)
    element_wise_xent = relu_preds - y_true * y_pred + sigmoid_logits
    return tf.reduce_sum(element_wise_xent)


def sigmoid(x):
    return 1/(1+np.exp(-x))


def softmax(x):
    x = x - x.max(axis=1).reshape(-1,1)
    e = np.exp(x)
    return e/e.sum(axis=1).reshape(-1,1)