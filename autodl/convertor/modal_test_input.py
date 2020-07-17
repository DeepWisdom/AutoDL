#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : wangjinlin (betterwang@fuzhi.ai)
# @Desc   : construct the AutoDLDataset to predict with Model

from typing import List
import os
import pandas as pd
import glob
from langdetect import detect as lang_detect

from autodl.auto_ingestion.dataset import AutoDLDataset, AutoDLMetadata
from autodl.convertor.image_to_tfrecords import get_features as image_get_features
from autodl.convertor.video_to_tfrecords import get_features as video_get_features
from autodl.convertor.speech_to_tfrecords import get_features as speech_get_features
from autodl.convertor.nlp_to_tfrecords import read_file as text_read_file, get_features as text_get_features

FILE_TYPES = {
    "image": [".jpg", ".jpeg", ".png", ".bmp", ".JPEG", ".JPG"],
    "video": [".avi"],
    "text": [".data"],
    "speech": [".wav"]
}


def collect_avaliable_files(input_dir, file_types: List):
    if not os.path.isdir(input_dir):
        raise IOError(f"{input_dir} is not a directory")

    avaliable_files = []
    for file in os.listdir(input_dir):
        if not os.path.isdir(file):
            file_abspath = os.path.join(input_dir, file)
            prefix, ext = os.path.splitext(file_abspath)
            if ext in file_types:
                avaliable_files.append(file)
    return avaliable_files


def check_test_data(test_data_path, modal_type="image"):
    """
    :desc 检测测试数据是否合法
    """
    if not os.path.isdir(test_data_path):
        raise NotADirectoryError("{} is not a directory!".format(test_data_path))
    labels_csv_files = [file for file in glob.glob(os.path.join(test_data_path, "*labels*.csv"))]
    if len(labels_csv_files) > 1:
        raise ValueError("Ambiguous label file! Several of them found: {}".format(labels_csv_files))
    elif len(labels_csv_files) == 0:
        label_df = None
    else:
        labels_csv_file = labels_csv_files[0]
        label_df = pd.read_csv(labels_csv_file)

        if modal_type == "text":
            if "Labels" not in label_df:
                raise ValueError("label file should contain `Labels` column")
        else:
            if "FileName" not in label_df or "Labels" not in label_df:
                raise ValueError("label file should contain `FileName` and `Labels` column")
    return label_df


def modal_test_input(input_dir, modal_type="image", vocabulary=None):
    language = "en"
    label_df = check_test_data(test_data_path=input_dir, modal_type=modal_type)

    all_avaliable_files = collect_avaliable_files(input_dir, file_types=FILE_TYPES.get(modal_type, []))

    if (label_df is not None) and (modal_type != "text"):
        avaliable_files = []
        labels = list(label_df["FileName"])
        for file in labels:
            if file in all_avaliable_files:
                avaliable_files.append(file)

    else:
        avaliable_files = all_avaliable_files
        if modal_type == "text":
            text_data_files = [file for file in glob.glob(os.path.join(input_dir, '*.data'))]
            if len(text_data_files) != 1:
                raise ValueError("No data file or Ambiguous files")

            text_data = text_read_file(text_data_files[0])
            data_str = " ".join(text_data)
            language = lang_detect(data_str)

    def func(file):
        if modal_type == "image":
            feature = image_get_features(dataset_dir=input_dir, filename=file)
            label = [0]  # mock data, useless
        elif modal_type == "video":
            feature = video_get_features(dataset_dir=input_dir, filename=file)
            label = [0]  # mock data, useless
        elif modal_type == "text":
            pass
        elif modal_type == "speech":
            feature = speech_get_features(dataset_dir=input_dir, filename=file)
            label = [0, 1]  # mock data, useless

        return feature, label

    def func_text(row):
        feature = text_get_features(row, vocabulary, language=language)
        label = [0, 1]  # mock data, useless
        return feature, label

    if modal_type == "text":
        features_labels_pairs = lambda: map(func_text, text_data)
    else:
        features_labels_pairs = lambda: map(func, avaliable_files)

    return features_labels_pairs, label_df
