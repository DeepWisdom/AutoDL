#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : run single image classification model with mnist dataset
# run example: python3 run_image_classification_example_without_tfrecords.py --input_data_path=../sample_data/monkeys/
# In this case: If you set `gen_tfrecords=False, gen_dataset=True`, the transformed data won't save into tfrecords file,
# and you can just use `data_formatter` return by `autoimage_2_autodl_format` to get train&test data.

import os
import time
import argparse

from autodl.convertor.image_to_tfrecords import autoimage_2_autodl_format
from autodl.auto_ingestion import data_io
from autodl.auto_models.auto_image.model import Model as ImageModel
from autodl.metrics.scores import autodl_auc, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="image example arguments")
    parser.add_argument("--input_data_path", type=str, help="path of input data")
    args = parser.parse_args()

    input_dir = os.path.dirname(args.input_data_path)

    data_formatter = autoimage_2_autodl_format(input_dir=input_dir, gen_tfrecords=False, gen_dataset=True)

    D_train = data_formatter.get_dataset_train()
    D_test = data_formatter.get_dataset_test()
    solution = data_formatter.get_test_solution()

    max_epoch = 50
    time_budget = 1200

    model = ImageModel(D_train.get_metadata())

    start_time = int(time.time())
    for i in range(max_epoch):
        remaining_time_budget = start_time + time_budget - int(time.time())
        model.fit(D_train.get_dataset(), remaining_time_budget=remaining_time_budget)

        remaining_time_budget = start_time + time_budget - int(time.time())
        y_pred = model.predict(D_test.get_dataset(), remaining_time_budget=remaining_time_budget)

        # Evaluation.
        nauc_score = autodl_auc(solution=solution, prediction=y_pred)
        acc_score = accuracy(solution=solution, prediction=y_pred)

        print("Epoch={}, evaluation: nauc_score={}, acc_score={}".format(i, nauc_score, acc_score))
