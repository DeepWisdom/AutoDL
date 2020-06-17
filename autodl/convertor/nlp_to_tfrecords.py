#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc: transform text format to tfrecords
# @Usage: `python3 nlp_to_tfrecords.py path/to/dataset`

# Input format files tree:
# ├── AutoNLP Dataset (name)
#     ├── labels.csv (data solution)
#     ├── name.data (each line is a string representing one example)

import os
import argparse
import json
from langdetect import detect as lang_detect

from autodl.utils.format_utils import *
from autodl.convertor.dataset_formatter import UniMediaDatasetFormatter


def read_file(filename):
    with open(filename, "r") as fin:
        output = fin.read().split("\n")
        if "" in output:
            output.remove("")
    return output


def clean_token(token):
    return repr(token)[1:-1]  # repr(repr(token))[2:-2]


def create_vocabulary(data, language="en"):
    print("Creating vocabulary...")
    vocabulary = {}
    i = 0
    for row in data:
        if language != "zh-cn":
            row = row.split(" ")
        for token in row:
            # Split (EN or ZH)
            cleaned_token = clean_token(token)
            if cleaned_token not in vocabulary:
                vocabulary[cleaned_token] = i
                i += 1
    return vocabulary


def get_features(row, vocabulary, language="en", format="DENSE"):
    """
    Args:
      row: string, a sentence in certain language (e.g. EN or ZH)
      vocabulary: dict, mapping from token to its index
      language: string, can be "EN" or "ZH"
    Returns:
      if DENSE format:
        a list of 4-tuples of form (row, col, channel)
      if SPARSE format:
        a list of 4-tuples of form (row_index, col_index, channel_index, value)
        for a sparse representation of a 3-D Tensor.
    """
    features = []
    if language != "zh-cn":
        row = row.split(" ")
    for e in row:
        token = clean_token(e)
        # if format=="DENSE":
        #     one_hot_word = [0]*len(vocabulary)
        #     one_hot_word[vocabulary[token]] = 1
        #     features.append(one_hot_word)
        # elif format=="SPARSE":
        #     features.append((0, 0, vocabulary[token], 1))
        features.append([vocabulary[token]])
        # else:
        #     raise Exception("Unknown format: {}".format(format))
    return features


def get_features_labels_pairs_nlp(data, merged_df, vocabulary, language, subset="train", label_format="DENSE"):
    labels_map = get_labels_map(merged_df)

    def func(x):
        row_index, row = x
        if "LabelConfidencePairs" in row:
            labels = row["LabelConfidencePairs"]
            confidence_pairs = True
        elif "Labels" in row:
            labels = row["Labels"]
            confidence_pairs = False
        else:
            raise Exception("No labels found, please check labels.csv file.")
        features = get_features(data[row_index], vocabulary=vocabulary, language=language)
        labels = get_labels(labels=labels, confidence_pairs=confidence_pairs, label_format=label_format,
                            labels_map=labels_map)
        return features, labels

    g = merged_df[merged_df["subset"] == subset].iterrows
    features_labels_pairs = lambda: map(func, g())
    return features_labels_pairs


def autonlp_2_autodl_format(input_dir: str, gen_tfrecords=True, gen_dataset=False, train_size=0.7):
    """
    There should be `labels.csv`, {basename}.data (each line is a string of one example separated by whitespace)
    labels.csv's data like:
        Labels   # head column name
        angry
        happy
        silence
    {basename}.data's data like:
        I'm sad for lost my dog
        yesterday I have a big deal

    usually, text language is important for model, here, we use langdetect to autodetect text language
    """
    input_dir = os.path.normpath(input_dir)
    name = os.path.basename(input_dir)
    output_dir = input_dir + "_formatted"

    labels_df = get_labels_df(input_dir, shuffling=False)  # Here shuffling should be False
    merged_df = get_merged_df(labels_df, train_size=train_size)
    all_classes = list(get_labels_map(merged_df).keys())

    # Read data
    all_data = read_file(os.path.join(input_dir, name + ".data"))
    data_str = " ".join(all_data)
    language = lang_detect(data_str)
    if language not in ["en", "zh-cn"]:
        raise ValueError("we detect language: {} in data file, "
                         "but so far we only support `en` and `zh-cn`".format(language))
    print("The data language detected is: {}".format(language))

    # Create vocabulary
    vocabulary = create_vocabulary(all_data, language)

    # Convert data into sequences of integers
    features_labels_pairs_train = get_features_labels_pairs_nlp(data=all_data, merged_df=merged_df,
                                                                vocabulary=vocabulary, language=language,
                                                                subset="train", label_format="DENSE")
    features_labels_pairs_test = get_features_labels_pairs_nlp(data=all_data, merged_df=merged_df,
                                                               vocabulary=vocabulary, language=language,
                                                               subset="test", label_format="DENSE")

    # Write data in TFRecords and vocabulary in metadata
    output_dim = len(all_classes)
    col_count, row_count = 1, 1
    sequence_size = -1
    num_channels = 1  # len(vocabulary)
    num_examples_train = merged_df[merged_df["subset"] == "train"].shape[0]
    num_examples_test = merged_df[merged_df["subset"] == "test"].shape[0]
    new_dataset_name = name  # same name
    classes_list = None
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
                                                 channels_dict=vocabulary,
                                                 gen_tfrecords=gen_tfrecords,
                                                 gen_dataset=gen_dataset)

    dataset_formatter.press_a_button_and_give_me_an_AutoDL_dataset()

    return dataset_formatter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", type=str, help="path of input data")
    args = parser.parse_args()

    input_dir = os.path.dirname(args.input_data_path)

    autonlp_2_autodl_format(input_dir=input_dir, gen_tfrecords=True, gen_dataset=False)
