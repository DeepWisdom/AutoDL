import tensorflow as tf
import numpy as np
import time

# from at_toolkit.at_utils import info, error, as_timer
from at_toolkit.interface.adl_tfds_convertor import AbsTfdsConvertor


class TfdsConvertor(AbsTfdsConvertor):
    def __init__(self, if_train_shuffle=False, train_shuffle_size=100, if_pad_batch=False, padded_batch_size=20, domain=None):
        self.train_tfds = None
        self.test_tfds = None
        self.train_num = 0
        self.test_num = 0
        self.accum_train_x = list()
        self.accum_train_y = None
        self.accm_train_cnt = 0
        self.accum_test_x = list()
        self.accum_test_y = list()
        self.accm_test_cnt = 0

        self.tfds_train_os_iterator = None
        self.tfds_train_iter_next = None

        self.speech_train_dataset = {"x": None, "y": None}
        self.speech_test_dataset = None
        self.speech_x_test = None
        self.if_train_shuffle = if_train_shuffle
        self.train_shuffle_size = train_shuffle_size
        self.train_max_shuffle_size = 1000
        self.if_padded_batch = if_pad_batch
        self.padded_batch_size = padded_batch_size

        self.tfds_convertor_sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

        self.domain = domain

    def init_train_tfds(self, train_tfds, train_num, force_shuffle=False):
        if self.train_tfds is None or self.train_num == 0 or force_shuffle is True:
            self.train_num = train_num
            self.train_tfds = train_tfds

            if self.if_train_shuffle or force_shuffle is True:
                self.train_tfds = self.train_tfds.shuffle(buffer_size=min(self.train_max_shuffle_size, int(self.train_num * 0.6)))

            if self.if_padded_batch:
                self.train_tfds = self.train_tfds.padded_batch(
                    self.padded_batch_size,
                    padded_shapes=([None, 1, 1, 1], [None]),
                    padding_values=(tf.constant(-1, dtype=tf.float32), tf.constant(-1, dtype=tf.float32)),
                )
            self.tfds_train_os_iterator = None

    def init_test_tfds(self, test_tfds):
        if self.test_tfds is None:
            self.test_tfds = test_tfds

            if self.if_padded_batch:
                self.test_tfds = test_tfds.padded_batch(
                    self.padded_batch_size,
                    padded_shapes=([None, 1, 1, 1], [None]),
                    padding_values=(tf.constant(-1, dtype=tf.float32), tf.constant(-1, dtype=tf.float32)),
                )

    def get_train_np(self, take_size):
        if self.train_tfds is None:
            return self.accum_train_x, self.accum_train_y

        if self.tfds_train_os_iterator is None:
            self.tfds_train_os_iterator = self.train_tfds.make_one_shot_iterator()
            self.tfds_train_iter_next = self.tfds_train_os_iterator.get_next()

        cur_get_cnt = 0
        cur_data_y = list()
        cur_incre_train_x = list()

        if self.accm_train_cnt < self.train_num:
            time_train_np_start = time.time()
            if self.if_padded_batch:
                while True:
                    example_batch_num = 0
                    try:
                        example, labels = self.tfds_convertor_sess.run(self.tfds_train_iter_next)
                        example = np.squeeze(example, (2, 3))
                        example = np.squeeze(example, axis=-1)
                        example = example.astype(np.int)
                        cur_incre_train_x.extend(example)
                        cur_data_y.extend(labels)
                        cur_get_cnt += example.shape[0]
                        self.accm_train_cnt += example.shape[0]
                        example_batch_num += 1

                        if cur_get_cnt >= take_size or self.accm_train_cnt >= self.train_num:
                            break

                    except tf.errors.OutOfRangeError:
                        break

            else:
                while True:
                    try:
                        example, labels = self.tfds_convertor_sess.run(self.tfds_train_iter_next)

                        cur_incre_train_x.append(example)
                        cur_data_y.append(labels)
                        cur_get_cnt += 1
                        self.accm_train_cnt += 1
                        if cur_get_cnt >= take_size or self.accm_train_cnt >= self.train_num:
                            break

                    except tf.errors.OutOfRangeError:
                        break

            self.accum_train_x.extend(cur_incre_train_x)

            if self.accum_train_y is None:
                self.accum_train_y = np.array(cur_data_y)
            else:
                self.accum_train_y = np.concatenate((self.accum_train_y, np.array(cur_data_y)))

        else:
            self.tfds_convertor_sess.close()

        return {"x": [np.squeeze(x) for x in cur_incre_train_x], "y": np.array(cur_data_y)}

    def get_train_np_accm(self, take_size) -> dict:
        self.get_train_np(take_size)
        return {"x": [np.squeeze(x) for x in self.accum_train_x], "y": np.array(self.accum_train_y)}

    def get_train_np_full(self):
        left_train_num = self.train_num - self.accm_train_cnt
        self.get_train_np(take_size=left_train_num)
        return {"x": [np.squeeze(x) for x in self.accum_train_x], "y": np.array(self.accum_train_y)}

    def get_test_np(self):
        if self.test_tfds is None:
            return self.accum_test_x, self.accum_test_y

        if len(self.accum_test_x) == 0:
            time_test_np_start = time.time()
            tfds_test_os_iterator = self.test_tfds.make_one_shot_iterator()
            tfds_test_iter_next = tfds_test_os_iterator.get_next()

            if self.if_padded_batch:
                while True:
                    try:
                        example, labels = self.tfds_convertor_sess.run(tfds_test_iter_next)
                        example = np.squeeze(example, (2, 3))
                        example = np.squeeze(example, axis=-1)
                        example = example.astype(np.int)

                        self.accum_test_x.extend(example)
                        self.accum_test_y.extend(labels)
                        self.accm_test_cnt += example.shape[0]
                    except tf.errors.OutOfRangeError:
                        break
            else:
                while True:
                    try:
                        example, labels = self.tfds_convertor_sess.run(tfds_test_iter_next)
                        self.accum_test_x.append(example)
                        self.accum_test_y.append(labels)
                        self.accm_test_cnt += 1

                    except tf.errors.OutOfRangeError:
                        break

            time_test_np_end = time.time()

            self.accum_test_y = np.array(self.accum_test_y)

        return [np.squeeze(x) for x in self.accum_test_x]


