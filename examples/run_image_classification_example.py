#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : run single image classification model with mnist dataset
# run example: python3 run_image.py --input_data_path=../sample_data/monkeys/

import os
import argparse

from autodl.convertor.image_to_tfrecords import autoimage_2_autodl_format
from autodl.auto_ingestion import data_io
from autodl.auto_ingestion.dataset import AutoDLDataset
from autodl.auto_models.auto_image.model import Model as ImageModel
from autodl.auto_ingestion.pure_model_run import run_single_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="image example arguments")
    parser.add_argument("--input_data_path", type=str, help="path of input data")
    args = parser.parse_args()

    input_dir = os.path.dirname(args.input_data_path)

    autoimage_2_autodl_format(input_dir=input_dir)

    new_dataset_dir = input_dir + "_formatted" + "/" + os.path.basename(input_dir)
    datanames = data_io.inventory_data(new_dataset_dir)
    basename = datanames[0]
    print("train_path: ", os.path.join(new_dataset_dir, basename, "train"))

    D_train = AutoDLDataset(os.path.join(new_dataset_dir, basename, "train"))

    max_epoch = 50
    time_budget = 1200

    model = ImageModel(D_train.get_metadata())

    run_single_model(model, new_dataset_dir, basename, time_budget, max_epoch)
