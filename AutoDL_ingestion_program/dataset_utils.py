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

"""Util functions to help parsing a Tensorflow dataset."""

import tensorflow as tf


def enforce_sequence_size(sample, sequence_size):
  """Takes a Sample as 4-D tensor and enfore the sequence size.

  The first dimension of the tensor represents the sequence length. Bundles
  will be added or removed at the end.

  Args:
    sample: 4-D tensor representing a Sample.
    sequence_size: int representing the maximum sequence length.
  Returns:
    The input 4-D tensor with added padds or removed bundles if it didn't
    respect the sequence_size.
  """
  pad_size = tf.maximum(sequence_size - tf.shape(sample)[0], 0)

  padded_sample = tf.pad(sample, ((0, pad_size), (0, 0), (0, 0), (0, 0)))

  sample = tf.slice(padded_sample, [0, 0, 0, 0], [sequence_size, -1, -1, -1])
  return sample


def decompress_image(compressed_image, num_channels=3):
  """Decode a JPEG compressed image into a 3-D float Tensor.

  TODO(andreamichi): Test this function.

  Args:
    compressed_image: string representing an image compressed as JPEG.
  Returns:
    3-D float Tensor with values ranging from [0, 1).
  """
  # Note that the resulting image contains an unknown height and width
  # that is set dynamically by decode_jpeg. The returned image
  # is a 3-D Tensor of uint8 [0, 255]. The third dimension is the channel.
  image = tf.image.decode_image(compressed_image, channels=num_channels)

  # Use float32 rather than uint8.
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  image.set_shape([None, None, num_channels])

  return image
