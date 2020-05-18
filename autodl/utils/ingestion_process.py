#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc    :

import os
import time
import numpy as np
from autodl.utils.time_utils import Timer
from autodl.utils.exception import TimeoutException, ModelAttrLackException, BadPredShapeException
from autodl.utils.log_utils import logger
from autodl.auto_ingestion import data_io
from autodl.auto_ingestion.dataset import AutoDLDataset
from autodl.auto_models.model import Model


def run_ingestion(dataset_dir, output_dir, start_info_share_dict, end_info_share_dict, prediction_share_dict):
    """
    Args:
        dataset_dir: dataset path
        output_dir: output path
        start_info_share_dict: Process shared dict, to pass start_info
        end_info_share_dict: Process shared dict, to pass end_info
        prediction_share_dict: Process shared dict, to pass each prediction
    """
    ingestion_success = True
    default_time_budget = 1200  # unit: second

    # find available dataset
    datanames = data_io.inventory_data(dataset_dir)
    datanames = [x for x in datanames if x.endswith(".data")]
    if len(datanames) != 1:
        raise ValueError("{} datasets found in dataset_dir={}!\n".format(len(datanames), dataset_dir) +
                         "Please put only ONE dataset under dataset_dir: {}.")

    basename = datanames[0]

    logger.info("************************************************")
    logger.info("******** Processing dataset " + basename[:-5].capitalize() + " ********")
    logger.info("************************************************")

    # begin to create train and test set
    logger.info("Reading training set and test set...")
    D_train = AutoDLDataset(os.path.join(dataset_dir, basename, "train"))
    D_test = AutoDLDataset(os.path.join(dataset_dir, basename, "test"))

    # get correct prediction shape
    num_examples_test = D_test.get_metadata().size()
    output_dim = D_test.get_metadata().get_output_size()
    correct_prediction_shape = (num_examples_test, output_dim)

    # install required packages and initialize model
    try:
        timer = Timer()
        timer.set(default_time_budget)
        with timer.time_limit("Initialization Model"):
            logger.info("Creating model...this process should not exceed 20min.")

            model = Model(D_train.get_metadata())  # The metadata of D_train and D_test only differ in sample_count
    except TimeoutException as e:
        logger.info("[-] Initialization phase exceeded time budget. Move to train/predict phase")
    except Exception as e:
        logger.error("Failed to initializing model.")
        logger.error("Encountered exception:\n" + str(e), exc_info=True)

    # train and test phrase, let's do it!
    logger.info("=" * 5 + " Start core part of ingestion program. " + "=" * 5)
    start = time.time()

    # pass basic info to scoring_process
    def write_start_info():
        ingestion_pid = os.getpid()
        task_name = basename.split(".")[0]

        start_info_share_dict["ingestion_pid"] = ingestion_pid
        start_info_share_dict["task_name"] = task_name
        start_info_share_dict["time_budget"] = default_time_budget
        start_info_share_dict["start_time"] = start

    write_start_info()

    try:
        # pre-check Model's attribute
        for attr in ["fit", "predict"]:
            if not hasattr(model, attr):
                raise ModelAttrLackException("Your model object doesn't have the method `{}`. "
                                             "Please implement it in model.py.".format(attr))

        use_done_training_attr = hasattr(model, "done_training")
        if not use_done_training_attr:
            logger.warning("Your model object doesn't have an attribute " +
                           "`done_training`. But this is necessary for ingestion " +
                           "program to know whether the model has done training " +
                           "and to decide whether to proceed more training. " +
                           "Please add this attribute to your model.")

        # keeping track of how many predictions are made
        prediction_order_number = 0

        while (not (use_done_training_attr and model.done_training)):
            # Train the model
            remaining_time_budget = start + default_time_budget - time.time()
            logger.info("Begin training the model...")
            model.fit(D_train.get_dataset(), remaining_time_budget=remaining_time_budget)
            logger.info("Finished training the model.")

            # Make predictions using the trained model
            remaining_time_budget = start + default_time_budget - time.time()
            logger.info("Begin testing the model by making predictions on test set...")
            Y_pred = model.predict(D_test.get_dataset(), remaining_time_budget=remaining_time_budget)
            logger.info("Finished making predictions.")

            if Y_pred is None:  # Stop train/predict process if Y_pred is None
                logger.info("The method model.test returned `None`. Stop train/predict process.")
                break
            else:  # Check if the prediction has good shape
                prediction_shape = tuple(Y_pred.shape)
                if prediction_shape != correct_prediction_shape:
                    raise BadPredShapeException("Bad prediction shape! Expected {} but got {}."
                                                .format(correct_prediction_shape, prediction_shape))

            # pass prediction timestamp scoring_process
            """
            format
            0: 1557269953.5586617
            1: 1557269956.012751
            """
            start_info_share_dict[prediction_order_number] = time.time()

            # pass current phrase prediction to scoring_process
            def write_current_prediction():
                pred_result = []
                for row in Y_pred:
                    if type(row) is not np.ndarray and type(row) is not list:
                        row = [row]
                    pred_result.append([float(val) for val in row])
                prediction_share_dict[prediction_order_number] = pred_result

            write_current_prediction()

            prediction_order_number += 1
            logger.info("[+] {0:d} predictions made, time spent so far {1:.2f} sec"
                        .format(prediction_order_number, time.time() - start))

            remaining_time_budget = start + default_time_budget - time.time()
            logger.info("[+] Time left {0:.2f} sec".format(remaining_time_budget))
            if remaining_time_budget <= 0:
                break

    except Exception as e:
        ingestion_success = False
        logger.info("Failed to run ingestion.")
        logger.error("Encountered exception:\n" + str(e), exc_info=True)

    # record process result to file
    end_time = time.time()
    overall_time_spent = end_time - start
    end_filename = "end.txt"
    with open(os.path.join(output_dir, end_filename), "w") as f:
        # write to end_info_share_dict
        end_info_share_dict["ingestion_duration"] = overall_time_spent
        end_info_share_dict["ingestion_success"] = int(ingestion_success)
        end_info_share_dict["end_time"] = end_time

        f.write("ingestion_duration: " + str(overall_time_spent) + "\n")
        f.write("ingestion_success: " + str(int(ingestion_success)) + "\n")
        f.write("end_time: " + str(end_time) + "\n")
        logger.info("Wrote the file {} marking the end of ingestion.".format(end_filename))

        if ingestion_success:
            logger.info("[+] Done. Ingestion program successfully terminated.")
            logger.info("[+] Overall time spent %5.2f sec " % overall_time_spent)
        else:
            logger.error("[-] Done, but encountered some errors during ingestion.")
            logger.error("[-] Overall time spent %5.2f sec " % overall_time_spent)

    logger.info("[Ingestion terminated]")
