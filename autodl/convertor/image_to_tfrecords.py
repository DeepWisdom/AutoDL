#!/usr/bin/env python  
# -*- coding: utf-8 -*-
# @Desc   : transform image format to tfrecords

import os
import argparse
from PIL import Image

from autodl.utils.format_utils import *
from autodl.convertor.dataset_formatter import UniMediaDatasetFormatter


def get_features(dataset_dir, filename):
    """ Read a file
    """
    filepath = os.path.join(dataset_dir, filename)
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    features = [[image_bytes]]
    return features


def get_features_labels_pairs(merged_df, dataset_dir, subset='train'):
    def func(x):
        index, row = x
        filename = row['FileName']
        if 'LabelConfidencePairs' in row:
            labels = row['LabelConfidencePairs']
            confidence_pairs = True
        elif 'Labels' in row:
            labels = row['Labels']
            confidence_pairs = False
        else:
            raise Exception('No labels found, please check labels.csv file.')
        features = get_features(dataset_dir, filename)  # read file
        labels = get_labels(labels, confidence_pairs=confidence_pairs)  # read labels
        return features, labels

    g = merged_df[merged_df['subset'] == subset].iterrows
    features_labels_pairs = lambda: map(func, g())
    return features_labels_pairs


def im_size(input_dir, filenames):
    """ Find images width and length
            -1 means not fixed size
    """
    s = set()
    for filename in filenames:
        file_path = os.path.join(input_dir, filename)
        if not os.path.exists(file_path):
            continue
        im = Image.open(os.path.join(input_dir, filename))
        s.add(im.size)
    if len(s) == 1:
        row_count, col_count = next(iter(s))
    else:
        row_count, col_count = -1, -1
    print('Images size: {} x {}\n'.format(row_count, col_count))
    return row_count, col_count


def autoimage_2_autodl_format(input_dir, num_channels=3):
    """
    there should be `labels.name`, `labels.csv`, and images under the input_dir.
    And the images should be better have same shape.
    `labels.name` contains the name of the label in each row, like
        apple
        banana
    `labels.csv` contains two columns(`FileName`, `Labels`). The `FileName` is the image name, The `Labels` is the label
    index in `labels.name`, like
        FileName   Labels
        fig1.png   1
        fig2.png   0
    """
    input_dir = os.path.abspath(input_dir)
    dir_name = os.path.basename(input_dir)
    output_dir = input_dir + '_formatted'

    labels_df = get_labels_df(input_dir)
    merged_df = get_merged_df(labels_df, train_size=0.7)
    all_classes = get_all_classes(merged_df)

    features_labels_pairs_train =\
        get_features_labels_pairs(merged_df, input_dir, subset='train')
    features_labels_pairs_test =\
        get_features_labels_pairs(merged_df, input_dir, subset='test')

    output_dim = len(all_classes)
    sequence_size = 1
    num_examples_train = merged_df[merged_df['subset'] == 'train'].shape[0]
    num_examples_test = merged_df[merged_df['subset'] == 'test'].shape[0]

    filenames = labels_df['FileName']
    row_count, col_count = im_size(input_dir, filenames)
    new_dataset_name = dir_name

    data_formatter = UniMediaDatasetFormatter(dataset_name=new_dataset_name,
                                              output_dir=output_dir,
                                              features_labels_pairs_train=features_labels_pairs_train,
                                              features_labels_pairs_test=features_labels_pairs_test,
                                              output_dim=output_dim,
                                              col_count=col_count,
                                              row_count=row_count,
                                              sequence_size=sequence_size,
                                              num_channels=num_channels,
                                              num_examples_train=num_examples_train,
                                              num_examples_test=num_examples_test,
                                              is_sequence_col="false",
                                              is_sequence_row="false",
                                              has_locality_col="true",
                                              has_locality_row="true",
                                              is_sequence="false",
                                              format="COMPRESSED",
                                              sequence_size_func=None,
                                              new_dataset_name=new_dataset_name,
                                              classes_list=None)
    data_formatter.press_a_button_and_give_me_an_AutoDL_dataset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", type=str, help="path of input data")
    args = parser.parse_args()

    input_dir = os.path.dirname(args.input_data_path)

    autoimage_2_autodl_format(input_dir=input_dir)
