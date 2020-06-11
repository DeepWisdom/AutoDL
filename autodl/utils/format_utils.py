#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : format func to format image, video etc.

import os
import pandas as pd
import numpy as np
import glob
from sklearn.utils import shuffle
from typing import Dict

from autodl.convertor.dataset_formatter import label_sparse_to_dense


def get_labels_df(dataset_dir, shuffling=True):
    """ Read labels.csv and return DataFrame
    """
    if not os.path.isdir(dataset_dir):
        raise IOError("{} is not a directory!".format(dataset_dir))
    labels_csv_files = [file for file in glob.glob(os.path.join(dataset_dir, '*labels*.csv'))]
    if len(labels_csv_files) > 1:
        raise ValueError("Ambiguous label file! Several of them found: {}".format(labels_csv_files))
    elif len(labels_csv_files) < 1:
        raise ValueError("No label file found! The name of this file should follow the glob "
                         "pattern `*labels*.csv` (e.g. monkeys_labels_file_format.csv).")
    else:
        labels_csv_file = labels_csv_files[0]
    labels_df = pd.read_csv(labels_csv_file)
    if shuffling:
        labels_df = shuffle(labels_df, random_state=42)
    return labels_df


def get_merged_df(labels_df, train_size=0.8):
    """Do train/test split (if needed) by generating random number in [0,1]."""
    merged_df = labels_df.copy()
    if 'subset' not in labels_df:
        np.random.seed(42)

        def get_subset(u):
            if u < train_size:
                return 'train'
            else:
                return 'test'
        merged_df['subset'] = merged_df.apply(lambda x: get_subset(np.random.rand()), axis=1)
    return merged_df


def get_labels(labels, confidence_pairs=False, label_format="SPARSE", labels_map={}, raw_labels=False):
    """Parse label confidence pairs into two lists of labels and confidence.
    Args:
        labels: string, of form `2 0.0001 9 0.48776 0 1.0`." or "2 9 0"
        confidence_pairs: True if labels are confidence pairs.
        label_format: return labels in one-hot/multi-hot format or not
        labels_map: {label: index}
        raw_labels: True if return raw labels
    """
    if isinstance(labels, str):
        l_split = labels.split(' ')
    else:
        l_split = [labels]

    labels = []
    if confidence_pairs:
        for i, x in enumerate(l_split):
            if i % 2 == 0:
                if isinstance(x, float):
                    x = int(x)
                labels.append(x)
        confidences = [float(x) for i, x in enumerate(l_split) if i % 2 == 1]
    else:
        for x in l_split:
            if x == x:
                if isinstance(x, float):
                    x = int(x)
                labels.append(x)
        confidences = [1 for _ in labels]

    if raw_labels:
        return labels

    if len(labels_map) > 0:
        labels = [labels_map.get(label) for label in labels]

    if label_format == "SPARSE":
        # return label, confidences like [1]
        return labels, confidences
    else:
        labels = label_sparse_to_dense(labels, len(labels_map))
        return labels


def get_labels_map(merged_df):
    all_labels = set()
    # collect all labels' name
    if "LabelConfidencePairs" in merged_df:
        labels = list(merged_df["LabelConfidencePairs"])
        confidence_pairs = True
    elif "Labels" in merged_df:
        labels = list(merged_df["Labels"])
        confidence_pairs = False
    else:
        raise Exception("No labels found, please check labels.csv file.")

    for label in labels:
        tmp_labels = get_labels(labels=label, confidence_pairs=confidence_pairs, raw_labels=True)
        all_labels.update(tmp_labels)

    all_labels = sorted(all_labels)
    labels_map = dict(zip(all_labels, range(len(all_labels))))
    return labels_map


def get_features_labels_pairs_from_rawdata(merged_df, dataset_dir, get_feature_func=None,
                                           subset="train", num_channels=3, label_format="SPARSE"):
    if get_feature_func is None:
        raise ValueError("get_feature_func should't be None")

    labels_map = get_labels_map(merged_df)

    def func(x):
        index, row = x
        filename = row["FileName"]
        if "LabelConfidencePairs" in row:
            labels = row["LabelConfidencePairs"]
            confidence_pairs = True
        elif "Labels" in row:
            labels = row["Labels"]
            confidence_pairs = False
        else:
            raise Exception("No labels found, please check labels.csv file.")
        features = get_feature_func(dataset_dir, filename, num_channels=num_channels)  # read file
        labels = get_labels(labels, confidence_pairs=confidence_pairs,
                            label_format=label_format, labels_map=labels_map)  # read labels
        return features, labels

    g = merged_df[merged_df["subset"] == subset].iterrows
    features_labels_pairs = lambda: map(func, g())
    return features_labels_pairs
