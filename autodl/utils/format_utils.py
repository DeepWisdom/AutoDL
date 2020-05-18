#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : format func to format image, video etc.

import os
import pandas as pd
import numpy as np
import glob
from sklearn.utils import shuffle


def get_labels_df(dataset_dir, shuffling=True):
    """ Read labels.csv and return DataFrame
    """
    if not os.path.isdir(dataset_dir):
        raise IOError("{} is not a directory!".format(dataset_dir))
    labels_csv_files = [file for file in glob.glob(os.path.join(dataset_dir, '*labels*.csv'))]
    if len(labels_csv_files) > 1:
        raise ValueError("Ambiguous label file! Several of them found: {}".format(labels_csv_files))
    elif len(labels_csv_files) < 1:
        raise ValueError("No label file found! The name of this file should follow the glob pattern `*labels*.csv` (e.g. monkeys_labels_file_format.csv).")
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


def get_labels(labels, confidence_pairs=False):
    """Parse label confidence pairs into two lists of labels and confidence.
    Args:
        labels: string, of form `2 0.0001 9 0.48776 0 1.0`." or "2 9 0"
        confidence_pairs: True if labels are confidence pairs.
    """
    if isinstance(labels, str):
        l_split = labels.split(' ')
    else:
        l_split = [labels]
    if confidence_pairs:
        labels = [int(x) for i, x in enumerate(l_split) if i % 2 == 0]
        confidences = [float(x) for i, x in enumerate(l_split) if i % 2 == 1]
    else:
        labels = [int(x) for x in l_split if x == x]  # x==x to remove NaN values
        confidences = [1 for _ in labels]
    return labels, confidences


def get_all_classes(merged_df):
    if 'LabelConfidencePairs' in list(merged_df):
        label_confidence_pairs = merged_df['LabelConfidencePairs']
        confidence_pairs = True
    elif 'Labels' in list(merged_df):
        label_confidence_pairs = merged_df['Labels']
        confidence_pairs = False
    else:
        raise Exception('No labels found, please check labels.csv file.')

    labels_sets = label_confidence_pairs.apply(lambda x: set(get_labels(x, confidence_pairs=confidence_pairs)[0]))
    all_classes = set()
    for labels_set in labels_sets:
        all_classes = all_classes.union(labels_set)
    return all_classes
