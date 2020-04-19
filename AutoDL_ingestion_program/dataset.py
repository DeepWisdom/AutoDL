# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AutoDL datasets.

Reads data in the Tensorflow AutoDL standard format.
"""
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
from google.protobuf import text_format
import dataset_utils
from data_pb2 import DataSpecification
from data_pb2 import MatrixSpec


def metadata_filename(dataset_name):
  return os.path.join("", dataset_name, "metadata.textproto")


def dataset_file_pattern(dataset_name):
  return os.path.join("", dataset_name, "sample*")


class AutoDLMetadata(object):
  """AutoDL data specification."""

  def __init__(self, dataset_name):
    self.dataset_name_ = dataset_name
    self.metadata_ = DataSpecification()
    with gfile.GFile(metadata_filename(dataset_name), "r") as f:
      text_format.Merge(f.read(), self.metadata_)

  def get_dataset_name(self):
    return self.dataset_name_

  def is_compressed(self, bundle_index=0):
    return self.metadata_.matrix_spec[
        bundle_index].format == MatrixSpec.COMPRESSED

  def is_sparse(self, bundle_index=0):
    return self.metadata_.matrix_spec[bundle_index].format == MatrixSpec.SPARSE

  def get_bundle_size(self):
    return len(self.metadata_.matrix_spec)

  def get_matrix_size(self, bundle_index=0):
    return (self.metadata_.matrix_spec[bundle_index].row_count,
            self.metadata_.matrix_spec[bundle_index].col_count)

  def get_num_channels(self, bundle_index=0):
    num_channels = self.metadata_.matrix_spec[bundle_index].num_channels
    if num_channels == -1: # Unknown or undefined num_channels
      if self.is_compressed(bundle_index): # If is compressed image, set to 3
        return 3
      else:
        return 1
    else:
      return num_channels

  def get_tensor_size(self, bundle_index=0):
    """For a dataset with examples of shape (T,H,W,C), return the shape (H,W,C).
    """
    matrix_size = self.get_matrix_size(bundle_index)
    num_channels = self.get_num_channels(bundle_index)
    return matrix_size[0], matrix_size[1], num_channels

  def get_tensor_shape(self, bundle_index=0):
    """ get_tensor_size updated with sequence size """
    sequence_size = self.get_sequence_size()
    row_count, col_count = self.get_matrix_size(bundle_index)
    num_channels = self.get_num_channels(bundle_index)
    return (sequence_size, row_count, col_count, num_channels)

  def get_sequence_size(self):
    return self.metadata_.sequence_size

  def get_output_size(self):
    return self.metadata_.output_dim

  def size(self):
    return self.metadata_.sample_count

  def get_label_to_index_map(self):
    return self.metadata_.label_to_index_map

  def get_channel_to_index_map(self):
    return self.metadata_.channel_to_index_map

  def get_feature_to_index_map(self):
    return self.metadata_.feature_to_index_map


class AutoDLDataset(object):
  """AutoDL Datasets out of TFRecords of SequenceExamples.

     See cs///experimental/autodl/export/tensorflow/README.md for more details
     on the features and labels.
  """

  def __init__(self, dataset_name, num_parallel_readers=3):
    """Construct an AutoDL Dataset.

    Args:
      dataset_name: name of the dataset under the 'dataset_dir' flag.
    """
    self.dataset_name_ = dataset_name
    self.num_parallel_readers = num_parallel_readers
    self.metadata_ = AutoDLMetadata(dataset_name)
    self._create_dataset()
    self.dataset_ = self.dataset_.map(self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def get_dataset(self):
    """Returns a tf.data.dataset object."""
    return self.dataset_

  def get_metadata(self):
    """Returns an AutoDLMetadata object."""
    return self.metadata_

  def _feature_key(self, index, feature_name):
    return str(index) + "_" + feature_name

  def _parse_function(self, sequence_example_proto):
    """Parse a SequenceExample in the AutoDL/TensorFlow format.

    Args:
      sequence_example_proto: a SequenceExample with "x_dense_input" or sparse
          input representation.
    Returns:
      An array of tensors. For first edition of AutoDl challenge, returns a
          pair `(features, labels)` where `features` is a Tensor of shape
            [sequence_size, row_count, col_count, num_channels]
          and `labels` a Tensor of shape
            [output_dim, ]
    """
    sequence_features = {}
    for i in range(self.metadata_.get_bundle_size()):
      if self.metadata_.is_sparse(i):
        sequence_features[self._feature_key(
            i, "sparse_col_index")] = tf.VarLenFeature(tf.int64)
        sequence_features[self._feature_key(
            i, "sparse_row_index")] = tf.VarLenFeature(tf.int64)
        sequence_features[self._feature_key(
            i, "sparse_channel_index")] = tf.VarLenFeature(tf.int64)
        sequence_features[self._feature_key(
            i, "sparse_value")] = tf.VarLenFeature(tf.float32)
      elif self.metadata_.is_compressed(i):
        sequence_features[self._feature_key(
            i, "compressed")] = tf.VarLenFeature(tf.string)
      else:
        sequence_features[self._feature_key(
            i, "dense_input")] = tf.FixedLenSequenceFeature(
                self.metadata_.get_tensor_size(i), dtype=tf.float32)
    # read TFRecord
    contexts, features = tf.parse_single_sequence_example(
        sequence_example_proto,
        context_features={
            "label_index": tf.VarLenFeature(tf.int64),
            "label_score": tf.VarLenFeature(tf.float32)
        },
        sequence_features=sequence_features)

    sample = [] # will contain [features, labels]
    for i in range(self.metadata_.get_bundle_size()):
      key_dense = self._feature_key(i, "dense_input")
      row_count, col_count = self.metadata_.get_matrix_size(i)
      num_channels = self.metadata_.get_num_channels(i)
      sequence_size = self.metadata_.get_sequence_size()
      fixed_matrix_size = row_count > 0 and col_count > 0
      row_count = row_count if row_count > 0 else None
      col_count = col_count if col_count > 0 else None
      if key_dense in features:
        f = features[key_dense]
        if not fixed_matrix_size:
          raise ValueError("To parse dense data, the tensor shape should " +
                           "be known but got {} instead..."\
                           .format((sequence_size, row_count, col_count)))
        f = tf.reshape(f, [sequence_size, row_count, col_count, num_channels])
        sample.append(f)

      sequence_size = sequence_size if sequence_size > 0 else None
      key_compressed = self._feature_key(i, "compressed")
      if key_compressed in features:
        compressed_images = features[key_compressed].values
        decompress_image_func =\
          lambda x: dataset_utils.decompress_image(x, num_channels=num_channels)
        # `images` here is a 4D-tensor of shape [T, H, W, C], some of which might be unknown
        images = tf.map_fn(
            decompress_image_func,
            compressed_images, dtype=tf.float32)
        images.set_shape([sequence_size, row_count, col_count, num_channels])
        sample.append(images)

      key_sparse_val = self._feature_key(i, "sparse_value")
      if key_sparse_val in features:
        key_sparse_col = self._feature_key(i, "sparse_col_index")
        key_sparse_row = self._feature_key(i, "sparse_row_index")
        key_sparse_channel = self._feature_key(i, "sparse_channel_index")
        sparse_col = features[key_sparse_col].values
        sparse_row = features[key_sparse_row].values
        try: # For back-compatibility. Before, there was no channel dimension.
          sparse_channel = features[key_sparse_channel].values
        except:
          # I think this won't work, Tensor object has no 'len'
          sparse_channel = [0] * len(sparse_col)
        sparse_val = features[key_sparse_val].values

        if col_count > num_channels:
            print('Sparse tabular data')
            # TABULAR: [120, 1]
            #          [1000, 2]
            #          [1504, 1]
            # each row is (index, value)
            sparse_col = tf.cast(sparse_col, tf.float32)
            sparse_channel = tf.cast(sparse_channel, tf.float32)
            tensor = tf.concat([
                tf.reshape(sparse_col, [-1, 1]),
                tf.reshape(sparse_val, [-1, 1])
                ], 1)
            tensor = tf.reshape(tensor, [1, -1, 2, 1])
            tensor = tf.cast(tensor, tf.float32)
            # Could use SparseTensor (to dense) because the shape of the dense tensor is known:
            # (1, col_count, 1, 1)
        else:
            print('Sparse text data')
            # TEXT: [232, 2, 41]
            # each example is a 'time series' of indexes
            tensor = tf.reshape(sparse_channel, [-1, 1, 1, 1])
            tensor = tf.cast(tensor, tf.float32)

        sample.append(tensor)
        # TODO: see how we can keep sparse tensors instead of
        # returning dense ones.

    label_indices = (contexts["label_index"].values,)
    label_indices = tf.reshape(label_indices, [-1, 1])
    sparse_tensor = tf.sparse.SparseTensor(indices=label_indices,
                                           values=contexts["label_score"].values,
                                           dense_shape=(self.metadata_.get_output_size(),))
    labels = tf.sparse.to_dense(sparse_tensor, validate_indices=False)
    sample.append(labels)
    return sample

  def _create_dataset(self):
    if not hasattr(self, "dataset_"):
      files = gfile.Glob(dataset_file_pattern(self.dataset_name_))
      if not files:
        raise IOError("Unable to find training files. data_pattern='" +
                      dataset_file_pattern(self.dataset_name_) + "'.")
      # logging.info("Number of training files: %s.", str(len(files)))
      if len(files) > 1:
        # Read in multiple tfrecord files and interleave them in parallel
        files = tf.data.Dataset.from_tensor_slices(files)
        dataset = files.interleave(
          tf.data.TFRecordDataset, cycle_length=self.num_parallel_readers,
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
      else:
        # Only a single tfrecord was given
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=self.num_parallel_readers)
      self.dataset_ = dataset

  def get_class_labels(self):
    """Get all class labels"""
    # -- IG: inefficient, but... not needed very often
    metadata=self.get_metadata()
    label_to_index_map = metadata.get_label_to_index_map()
    classes_list = [None] * len(label_to_index_map)
    for label in label_to_index_map:
      index = label_to_index_map[label]
      classes_list[index] = label
    return classes_list

  def get_nth_element(self, num):
    """Get n-th element in `autodl_dataset` using iterator."""
    # -- IG: replaced previous 3d version
    dataset = self.get_dataset()
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
      for _ in range(num+1):
        tensor_4d, labels = sess.run(next_element)
    return tensor_4d, labels

  def show_image(self, num):
    """Visualize a image represented by `tensor_4d` in RGB or grayscale."""
    # -- IG: replaced previous 3d version
    tensor_4d, label_confidence_pairs = self.get_nth_element(num)
    num_channels = tensor_4d.shape[-1]
    image = np.squeeze(tensor_4d[0])
    # If the entries are float but in [0,255]
    if not np.issubdtype(image.dtype, np.integer) and np.max(image) > 100:
      image = image / 256
    if num_channels == 1:
      plt.imshow(image, cmap='gray')
    else:
      # if not num_channels == 3:
      #   raise ValueError("Expected num_channels = 3 but got {} instead."\
      #                    .format(num_channels))
      plt.imshow(image)
    labels = self.get_class_labels()
    if labels:
      lbl_list = [lbl for lbl, pos in zip(labels, label_confidence_pairs) if pos]
      plt.title('Label(s): ' + ' '.join(lbl_list))
    else:
      plt.title('Labels: ' + str(label_confidence_pairs))
    plt.show()
    return plt

def main(argv):
  del argv  # Unused.
  dataset = AutoDLDataset("mnist")
  # dataset.init()
  iterator = dataset.get_dataset().make_one_shot_iterator()
  next_element = iterator.get_next()

  sess = tf.Session()
  for idx in range(10):
    print("Example " + str(idx))
    print(sess.run(next_element))


if __name__ == "__main__":
  app.run(main)
