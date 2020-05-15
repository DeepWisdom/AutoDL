# -*- coding: utf-8 -*-
from __future__ import absolute_import
import sys
import os
import logging
import tensorflow as tf
import torch


def get_logger(name, stream=sys.stderr, file_path='debug.log'):
    formatter = logging.Formatter(fmt='[%(asctime)s %(levelname)s %(filename)s] %(message)s')

    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    level = logging.INFO if os.environ.get('LOG_LEVEL', 'INFO') == 'INFO' else logging.DEBUG
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def get_tf_resize(height, width, times=1, min_value=0.0, max_value=1.0):
    def preprocessor(tensor):
        in_times, in_height, in_width, in_channels = tensor.get_shape()

        if width == in_width and height == in_height:
            pass
        else:
            tensor = tf.image.resize_images(tensor, (height, width), method=tf.image.ResizeMethod.BICUBIC)

        if times != in_times or times > 1:
            tensor = tf.reshape(tensor, [-1, height * width, in_channels])
            tensor = tf.image.resize_images(tensor, (times, height * width),
                                            method=tf.image.ResizeMethod.BICUBIC)
            tensor = tf.reshape(tensor, [times, height, width, in_channels])
        if times == 1:
            tensor = tensor[int(times // 2)]

        delta = max_value - min_value
        if delta < 0.9 or delta > 1.1 or min_value < -0.1 or min_value > 0.1:
            tensor = (tensor - min_value) / delta
        return tensor

    return preprocessor


def get_tf_to_tensor(is_random_flip=True):
    def preprocessor(tensor):
        if is_random_flip:
            tensor = tf.image.random_flip_left_right(tensor)
        dims = len(tensor.shape)
        if dims == 3:
            tensor = tf.transpose(tensor, perm=[2, 0, 1])
        elif dims == 4:
            tensor = tf.transpose(tensor, perm=[3, 0, 1, 2])
        return tensor

    return preprocessor


def crop(img, top, left, height, width):
    bs, c = img.shape[0], img.shape[1]
    new_img = torch.Tensor(bs, c, height, width).cuda()
    for i in range(bs):
        new_img[i] = img[i][..., top:top + height, left:left + width]
    return new_img


def center_crop(img, output_size):
    _, _, image_width, image_height = img.size()
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))

    return crop(img, crop_top, crop_left, crop_height, crop_width)


def five_crop(img, size):
    assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    _, _, image_height, image_width = img.size()
    crop_height, crop_width = size
    if crop_width > image_width or crop_height > image_height:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_height, image_width)))

    tl = crop(img, 0, 0, crop_height, crop_width)
    tr = crop(img, 0, image_width - crop_width, crop_height, crop_width)
    bl = crop(img, image_height - crop_height, 0, crop_height, crop_width)
    br = crop(img, image_height - crop_height, image_width - crop_width, crop_height, crop_width)
    center = center_crop(img, (crop_height, crop_width))

    return torch.cat([tl, tr, bl, br, center])
