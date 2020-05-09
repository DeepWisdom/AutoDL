#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc    : the entry of competition local test

import os
import argparse
import time
import numpy as np
from multiprocessing import Process, Manager

from autodl.utils.logger import logger
from autodl.utils.time_utils import Timer
from autodl.utils.exception import TimeoutException, ModelAttrLackException, BadPredShapeException
from autodl.auto_ingestion import data_io
from autodl.metrics.scores import autodl_auc, accuracy
from autodl.auto_ingestion.dataset import AutoDLDataset
from autodl.auto_models.model import Model
from autodl.auto_scoring.score_evaluator import ScoreEvaluator


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
    logger.info("******** Processing dataset " + basename[:-5].capitalize() + "********")
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

    # pre-check Model's attribute
    for attr in ["train", "test"]:
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
        # keeping track of how many predictions are made
        prediction_order_number = 0

        while (not (use_done_training_attr and model.done_training)):
            # Train the model
            remaining_time_budget = start + default_time_budget - time.time()
            logger.info("Begin training the model...")
            model.train(D_train.get_dataset(), remaining_time_budget=remaining_time_budget)
            logger.info("Finished training the model.")

            # Make predictions using the trained model
            remaining_time_budget = start + default_time_budget - time.time()
            logger.info("Begin testing the model by making predictions on test set...")
            Y_pred = model.test(D_test.get_dataset(), remaining_time_budget=remaining_time_budget)
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


def run_scoring(dataset_dir, output_dir, start_info_share_dict, end_info_share_dict, prediction_share_dict):
    """
    Args:
        dataset_dir: dataset path
        output_dir: output path
        start_info_share_dict: Process shared dict, start_info from run_ingestion
        end_info_share_dict: Process shared dict, end_info from run_ingestion
        prediction_share_dict: Process shared dict, each prediction from run_ingestion
    """
    logger.info("=" * 5 + " Start scoring program. " + "=" * 5)

    scoring_functions = {
        "nauc": autodl_auc,
        "accuracy": accuracy
    }

    evaluator = ScoreEvaluator(solution_dir=dataset_dir, score_dir=output_dir, scoring_functions=scoring_functions,
                               start_info_share_dict=start_info_share_dict,
                               end_info_share_dict=end_info_share_dict,
                               prediction_share_dict=prediction_share_dict)

    ingestion_start = evaluator.ingestion_start
    time_budget = evaluator.time_budget

    try:
        while (time.time() < ingestion_start + time_budget):
            if evaluator.ingestion_finished():
                logger.info("Detected ingestion program had stopped running. Stop scoring now.")
                evaluator.scoring_success = True
                break
            time.sleep(1)

            # Fetch new predictions, compute their scores and update variables #
            evaluator.score_new_predictions()

            logger.debug("Prediction files so far: {}".format(evaluator.prediction_list_so_far))
        else:
            # When time budget is used up, kill ingestion
            if evaluator.ingestion_is_alive():
                evaluator.time_limit_exceeded = True
                evaluator.kill_ingestion()
                logger.info("Detected time budget is used up. Killed ingestion and terminating scoring...")
    except Exception as e:
        evaluator.scoring_success = False
        logger.error("[-] Error occurred in scoring:\n" + str(e), exc_info=True)

    evaluator.score_new_predictions()

    logger.info("Final area under learning curve for {}: {:.4f}"
                .format(evaluator.task_name, evaluator.learning_curve.get_alc()))

    # Compute scoring error bars of last prediction
    n = 10
    logger.info("Computing error bars with {} scorings...".format(n))
    mean, std, var = evaluator.compute_error_bars(n=n)
    logger.info("\nLatest prediction NAUC:\n* Mean: {}\n* Standard deviation: {}\n* "
                "Variance: {}".format(mean, std, var))

    # Compute ALC error bars
    n = 5
    logger.info("Computing ALC error bars with {} curves...".format(n))
    mean, std, var = evaluator.compute_alc_error_bars(n=n)
    logger.info("\nArea under Learning Curve:\n* Mean: {}\n* Standard deviation: {}\n* "
                "Variance: {}".format(mean, std, var))

    scoring_start = evaluator.start_time
    if evaluator.scoring_success is not None and not evaluator.scoring_success:
        logger.error("[-] Some error occurred in scoring program. "
                     "Please see output/error log of Scoring Step.")
    elif not len(start_info_share_dict) > 0:
        # TODO 通过共享变量判断
        if evaluator.time_limit_exceeded:
            logger.error("[-] Ingestion program exceeded time budget. "
                         "Predictions made so far will be used for evaluation.")
        else:  # Less probable to fall in this case
            if evaluator.ingestion_is_alive():
                evaluator.kill_ingestion()
            logger.error("[-] No 'end.txt' file is produced by ingestion. " +
                         "Ingestion or scoring may have not terminated normally.")
    else:
        ingestion_duration = end_info_share_dict["ingestion_duration"]

        if end_info_share_dict["ingestion_success"] == 0:
            logger.error("[-] Some error occurred in ingestion program. " +
                         "Please see output/error log of Ingestion Step.")
        else:
            logger.info("[+] Successfully finished scoring! Scoring duration: {:.2f} sec. "
                        .format(time.time() - scoring_start) +
                        "Ingestion duration: {:.2f} sec. ".format(ingestion_duration) +
                        "The score of your algorithm on the task '{}' is: {:.6f}."
                        .format(evaluator.task_name, evaluator.learning_curve.get_alc()))

    logger.info("[Scoring terminated]")


def get_parser():
    parser = argparse.ArgumentParser()

    cur_path = os.path.dirname(os.path.abspath(__file__))
    default_dataset_dir = os.path.join(cur_path, "../sample_data/adult")
    default_output_dir = os.path.join(cur_path, "output")
    parser.add_argument("--dataset_dir", type=str, default=default_dataset_dir,
                        help="Directory storing the dataset (containing e.g. adult.data/)")
    parser.add_argument("--output_dir", type=str, default=default_output_dir,
                        help="Directory storing the predictions. It will contain e.g. "
                             "[start.txt, adult.predict_0, adult.predict_1, ..., end.txt] when ingestion terminates.")
    args = parser.parse_args()
    logger.info("Parsed args are: {}".format(str(args)))
    return args


def main():
    args = get_parser()
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    # create folder if not exist
    data_io.mkdir(output_dir)

    logger.info("#" * 50)
    logger.info("Begin running local test")
    logger.info("dataset_dir = {}".format(data_io.get_basename(dataset_dir)))
    logger.info("output_dir = {}".format(data_io.get_basename(output_dir)))
    logger.info("#" * 50)

    # create process sharing value
    with Manager() as manager:
        start_info_share_dict = manager.dict()
        end_info_share_dict = manager.dict()
        prediction_share_dict = manager.dict()

        # create process
        ingestion_process = Process(target=run_ingestion, name="ingestion", args=(dataset_dir, output_dir,
                                                                                  start_info_share_dict,
                                                                                  end_info_share_dict,
                                                                                  prediction_share_dict))
        scoring_process = Process(target=run_scoring, name="scoring", args=(dataset_dir, output_dir,
                                                                            start_info_share_dict,
                                                                            end_info_share_dict,
                                                                            prediction_share_dict))

        # clean path
        data_io.copy_dir(output_dir)
        data_io.remove_dir(output_dir)
        data_io.mkdir(output_dir)

        # start to run process
        ingestion_process.start()
        scoring_process.start()

        ingestion_process.join()
        scoring_process.join()

        if not ingestion_process.exitcode == 0:
            logger.warning("Some error occurred in ingestion program.")
        if not scoring_process.exitcode == 0:
            raise Exception("Some error occurred in scoring program.")


if __name__ == "__main__":
    main()
