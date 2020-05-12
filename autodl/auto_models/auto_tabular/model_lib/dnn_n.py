import numpy as np
import tensorflow as tf
from ..CONSTANT import *
from ..utils.data_utils import ohe2cat
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
        self.type = 'nn'
        self.patience = 10

        self.not_gain_threhlod = 3

    def init_model(self,
                   num_classes,
                   **kwargs):
        self.num_classes = num_classes
        model_fn = self.model_fn
        self._model = tf.estimator.Estimator(model_fn=model_fn)
        self.is_init = True

    def epoch_train(self, dataloader, epoch):
        dataset_train = dataloader['train']
        train_input_fn = lambda: self.input_function(dataset_train, 'train')
        self._model.train(input_fn=train_input_fn, steps=100)

    def epoch_valid(self, dataloader):
        dataset_valid, label = dataloader['valid']
        valid_input_fn = lambda: self.input_function(dataset_valid, 'valid')
        valid_results = self._model.predict(input_fn=valid_input_fn)
        preds = [x['probabilities'] for x in valid_results]
        preds = np.array(preds)
        valid_auc = roc_auc_score(label, preds)
        return valid_auc

    def predict(self, dataloader, batch_size=32):
        dataset_test = dataloader['test']
        valid_input_fn = lambda: self.input_function(dataset_test, 'test')
        test_results = self._model.predict(input_fn=valid_input_fn)
        preds = [x['probabilities'] for x in test_results]
        preds = np.array(preds)
        return preds

    def model_fn(self, features, labels, mode):
        is_training = False
        keep_prob = 1
        if mode == tf.estimator.ModeKeys.TRAIN:
            is_training = True
            keep_prob = 0.8

        input_layer = features

        # Replace missing values by 0
        mask = tf.is_nan(input_layer)
        input_layer = tf.where(mask, tf.zeros_like(input_layer), input_layer)

        # Sum over time axis
        input_layer = tf.reduce_sum(input_layer, axis=1)
        mask = tf.reduce_sum(1 - tf.cast(mask, tf.float32), axis=1)

        # Flatten
        input_layer = tf.layers.flatten(input_layer)
        mask = tf.layers.flatten(mask)
        f = input_layer.get_shape().as_list()[1]  # tf.shape(input_layer)[1]

        # Build network
        x = tf.layers.batch_normalization(input_layer, training=is_training)
        x = tf.nn.dropout(x, keep_prob)
        x_skip = self.fc(x, 256, is_training)
        x = self.fc(x_skip, 256, is_training)
        x = tf.nn.dropout(x, keep_prob)
        x = self.fc(x, 256, is_training) + x_skip
        x_mid = self.fc(x, f, is_training)

        x = self.fc(x_mid, 256, is_training)
        for i in range(3):
            x = self.fc(x, 256, is_training)

        logits = tf.layers.dense(inputs=x, units=self.num_classes)
        sigmoid_tensor = tf.nn.sigmoid(logits, name="sigmoid_tensor")

        predictions = {

            "classes": sigmoid_tensor > 0.5,

            "probabilities": sigmoid_tensor
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        loss_labels = tf.reduce_sum(sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
        loss_reconst = tf.reduce_sum(mask * tf.abs(tf.subtract(input_layer, x_mid)))
        loss = loss_labels + loss_reconst

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        assert mode == tf.estimator.ModeKeys.EVAL
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    def fc(self, x, out_dim, is_training):
        x = tf.layers.dense(inputs=x, units=out_dim)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        return x

    def input_function(self, dataset, mode):
        if mode == 'train':
            dataset = dataset.shuffle(buffer_size=100)
            dataset = dataset.repeat()

        dataset = dataset.batch(batch_size=128)

        iterator_name = 'iterator_' + mode

        if not hasattr(self, iterator_name):
            self.iterator = dataset.make_one_shot_iterator()
        iterator = dataset.make_one_shot_iterator()
        if mode == 'train':
            example, labels = iterator.get_next()
            return example, labels
        if mode == 'valid' or mode == 'test':
            example = iterator.get_next()
            return example


def sigmoid_cross_entropy_with_logits(labels, logits):
    labels = tf.cast(labels, dtype=tf.float32)
    relu_logits = tf.nn.relu(logits)
    exp_logits = tf.exp(- tf.abs(logits))
    sigmoid_logits = tf.log(1 + exp_logits)
    element_wise_xent = relu_logits - labels * logits + sigmoid_logits
    return tf.reduce_sum(element_wise_xent)