#!/usr/bin/env python  
# -*- coding: utf-8 -*-
# @Desc   : transform tabular format to tfrecords

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from autodl.convertor.dataset_formatter import UniMediaDatasetFormatter


def get_features(row):
    return [list(row)]


def get_labels(row):
    if isinstance(row, list) or isinstance(row, tuple) or isinstance(row, np.ndarray):
        return list(row)
    else:
        return [row]


def get_features_labels_pairs(data, solution):
    data.reset_index(drop=True, inplace=True)

    def func(i):
        features = get_features(data.iloc[i])
        labels = get_labels(solution[i])
        return features, labels

    g = iter(range(len(data)))
    features_labels_pairs = lambda: map(func, g)
    return features_labels_pairs


def autotabular_2_autodl_format(input_dir, data: pd.DataFrame, label: pd.Series):
    """
    Args:
        input_dir: dir of original tabular csv file
        data: input data
        label: input data label
    Returns:
    """
    input_dir = os.path.abspath(input_dir)
    dir_name = os.path.basename(input_dir)
    output_dir = input_dir + "_formatted"

    if label.name in data:
        del data[label.name]

    label = OneHotEncoder().fit_transform([[ele] for ele in label]).toarray()
    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.3, random_state=1024)

    # convert data into sequences
    features_labels_pairs_train = get_features_labels_pairs(train_data, train_label)
    features_labels_pairs_test = get_features_labels_pairs(test_data, test_label)

    # write data into TFRecords format
    output_dim = int(np.max(label)) + 1
    col_count = data.shape[1]
    row_count = 1
    sequence_size = 1
    num_channels = None
    num_examples_train = len(train_data)
    num_examples_test = len(test_data)
    new_dataset_name = dir_name
    classes_list = None
    channels_list = None

    dataset_formatter = UniMediaDatasetFormatter(dataset_name=new_dataset_name,
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
                                                 format="DENSE",
                                                 label_format="DENSE",
                                                 is_sequence="false",
                                                 sequence_size_func=None,
                                                 new_dataset_name=new_dataset_name,
                                                 classes_list=classes_list,
                                                 channels_list=channels_list)

    dataset_formatter.press_a_button_and_give_me_an_AutoDL_dataset()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", type=str, help="path of input data")
    args = parser.parse_args()

    input_dir = os.path.dirname(args.input_data_path)

    df = pd.read_csv(args.input_data_path, sep=";")

    trans_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week",
                  "poutcome", "y"]
    for col in trans_cols:
        lbe = LabelEncoder()
        df[col] = lbe.fit_transform(df[col])

    label = df["y"]

    autotabular_2_autodl_format(input_dir=input_dir, data=df, label=label)
