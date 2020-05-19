#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : run single tabular classification model with kaggle bank-additional dataset
# run example: python3 run_tabular_classification_example.py --input_data_path=../sample_data/bank/bank-additional-full.csv

import os
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from autodl.auto_ingestion import data_io
from autodl.auto_ingestion.dataset import AutoDLDataset
from autodl.convertor.tabular_to_tfrecords import autotabular_2_autodl_format
from autodl.auto_models.auto_tabular.model import Model as TabularModel
from autodl.auto_ingestion.pure_model_run import run_single_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tabular example arguments")
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

    new_dataset_dir = input_dir + "_formatted" + "/" + os.path.basename(input_dir)
    datanames = data_io.inventory_data(new_dataset_dir)
    basename = datanames[0]
    print("train_path: ", os.path.join(new_dataset_dir, basename, "train"))

    D_train = AutoDLDataset(os.path.join(new_dataset_dir, basename, "train"))
    D_test = AutoDLDataset(os.path.join(new_dataset_dir, basename, "test"))

    max_epoch = 50
    time_budget = 1200
    model = TabularModel(D_train.get_metadata())

    run_single_model(model, new_dataset_dir, basename, time_budget, max_epoch)
