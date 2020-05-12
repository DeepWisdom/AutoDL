#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc    :

import time
from autodl.utils.log_utils import logger

from autodl.metrics.scores import autodl_auc, accuracy
from autodl.auto_scoring.score_evaluator import ScoreEvaluator


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

    evaluator.activate_save_final()
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