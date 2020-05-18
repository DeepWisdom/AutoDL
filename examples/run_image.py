#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : run single image classification model with mnist dataset
# run example: python3 run_image.py --input_data_path=../sample_data/monkeys/

import os
import argparse
import time

from autodl.convertor.image_to_tfrecords import autoimage_2_autodl_format
from autodl.auto_ingestion import data_io
from autodl.utils.util import get_solution
from autodl.metrics import autodl_auc
from autodl.auto_ingestion.dataset import AutoDLDataset
from autodl.auto_models.auto_image.model import Model as ImageModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tabular example arguments")
    parser.add_argument("--input_data_path", type=str, help="path of input data")
    args = parser.parse_args()

    input_dir = os.path.dirname(args.input_data_path)

    autoimage_2_autodl_format(input_dir=input_dir)

    new_dataset_dir = input_dir + "_formatted" + "/" + os.path.basename(input_dir)
    datanames = data_io.inventory_data(new_dataset_dir)
    basename = datanames[0]
    print("train_path: ", os.path.join(new_dataset_dir, basename, "train"))

    D_train = AutoDLDataset(os.path.join(new_dataset_dir, basename, "train"))
    D_test = AutoDLDataset(os.path.join(new_dataset_dir, basename, "test"))

    max_epoch = 100
    time_budget = 1200
    start_time = int(time.time())

    model = ImageModel(D_train.get_metadata())

    for i in range(max_epoch):
        remaining_time_budget = start_time + time_budget - int(time.time())
        model.train(D_train.get_dataset(), remaining_time_budget=remaining_time_budget)

        remaining_time_budget = start_time + time_budget - int(time.time())
        y_pred = model.test(D_test.get_dataset(), remaining_time_budget=remaining_time_budget)

        solution = get_solution(new_dataset_dir)

        nauc = autodl_auc(solution, y_pred)
        print(f"epoch: {i}, nauc: {nauc}")
