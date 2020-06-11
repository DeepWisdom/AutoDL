#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : transform speech format to tfrecords

import os
from sys import argv
import librosa

from autodl.utils.format_utils import *
from autodl.convertor.dataset_formatter import UniMediaDatasetFormatter


def get_features(dataset_dir, filename, num_channels=1):
    filepath = os.path.join(dataset_dir, filename)
    y, sr = librosa.load(path=filepath, sr=None)
    features = [[e] for e in y]
    return features


def autospeech_2_autodl_format(input_dir: str, gen_tfrecords=True, gen_dataset=False, train_size=0.7):
    input_dir = os.path.normpath(input_dir)
    name = os.path.basename(input_dir)
    output_dir = input_dir + "_formatted"

    labels_df = get_labels_df(input_dir)
    merged_df = get_merged_df(labels_df, train_size=train_size)
    all_classes = list(get_labels_map(merged_df).keys())

    # Convert data into sequences of integers
    # Becase speech is in `Dense` format, set `label_format="DENSE"`, the format of label is in one-hot style
    # like label = [0, 0, 0, 1, 0, 0, 0]
    features_labels_pairs_train = get_features_labels_pairs_from_rawdata(merged_df, input_dir,
                                                                         get_feature_func=get_features,
                                                                         subset="train", num_channels=1,
                                                                         label_format="DENSE")
    features_labels_pairs_test = get_features_labels_pairs_from_rawdata(merged_df, input_dir,
                                                                        get_feature_func=get_features,
                                                                        subset="test", num_channels=1,
                                                                        label_format="DENSE")

    # Write data in TFRecords format
    output_dim = len(all_classes)
    col_count, row_count = 1, 1
    sequence_size = -1
    num_channels = 1
    num_examples_train = merged_df[merged_df["subset"] == "train"].shape[0]
    num_examples_test = merged_df[merged_df["subset"] == "test"].shape[0]
    new_dataset_name = name  # same name
    classes_list = None
    channels_list = None
    dataset_formatter = UniMediaDatasetFormatter(name,
                                                 output_dir,
                                                 features_labels_pairs_train,
                                                 features_labels_pairs_test,
                                                 output_dim,
                                                 col_count,
                                                 row_count,
                                                 sequence_size=sequence_size,  # for strides=2
                                                 num_channels=num_channels,
                                                 num_examples_train=num_examples_train,
                                                 num_examples_test=num_examples_test,
                                                 is_sequence_col="false",
                                                 is_sequence_row="false",
                                                 has_locality_col="true",
                                                 has_locality_row="true",
                                                 format="DENSE",
                                                 label_format="DENSE",
                                                 is_sequence="false",
                                                 sequence_size_func=None,
                                                 new_dataset_name=new_dataset_name,
                                                 classes_list=classes_list,
                                                 channels_list=channels_list,
                                                 gen_dataset=gen_dataset,
                                                 gen_tfrecords=gen_tfrecords)

    dataset_formatter.press_a_button_and_give_me_an_AutoDL_dataset()
    return dataset_formatter


if __name__ == "__main__":

    if len(argv) == 2:
        input_dir = argv[1]
    else:
        print("Please enter a dataset directory. Usage: `python3 speech_to_tfrecords path/to/dataset`")
        input_dir = None
        exit()

    autospeech_2_autodl_format(input_dir=input_dir)
