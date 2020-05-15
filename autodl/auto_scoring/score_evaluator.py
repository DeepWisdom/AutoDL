#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc    : evaluator to dynamic generate metric

import os
import time
import base64
import numpy as np
import matplotlib.pyplot as plt

from autodl.utils.task_type import is_multiclass
from autodl.utils.plot_alc import PlotAlc
from autodl.utils.log_utils import logger
from autodl.auto_scoring.libscores import ls
from autodl.utils.util import is_process_alive, terminate_process, get_solution
from autodl.utils.exception import IngestionException
from autodl.utils.learning_curve import LearningCurve


class ScoreEvaluator(object):

    def __init__(self, solution_dir=None, prediction_dir=None, score_dir=None,
                 scoring_functions=None, task_name=None, participant_name=None,
                 algorithm_name=None, submission_id=None,
                 start_info_share_dict={},
                 end_info_share_dict={},
                 prediction_share_dict={}):
        """
        Args:
          scoring_functions: a dict containing (string, scoring_function) pairs
        """
        self.start_time = time.time()

        # shared variable throw Process
        self.start_info_share_dict = start_info_share_dict
        self.end_info_share_dict = end_info_share_dict
        self.prediction_share_dict = prediction_share_dict

        self.solution_dir = solution_dir
        self.prediction_dir = prediction_dir
        self.score_dir = score_dir
        self.scoring_functions = scoring_functions

        self.task_name = task_name or self.get_task_name(solution_dir)
        self.participant_name = participant_name
        self.algorithm_name = algorithm_name
        self.submission_id = submission_id

        # State variables
        self.scoring_success = None
        self.time_limit_exceeded = None
        self.scores_so_far = {"nauc": []}
        self.relative_timestamps = []
        # new variable
        self.prediction_list_so_far = []  # store all prediction order_number
        self.new_prediction_list = []  # store newest prediction order_number

        # ingestion variable
        self.ingestion_info = {}
        self.ingestion_start = 0
        self.ingestion_pid = 0
        self.time_budget = 0

        # save final
        self.save_final = False  # call activate_save_final to save figure

        # Resolve info from directories
        self.solution = get_solution(self.solution_dir)
        # Check if the task is multilabel (i.e. with one hot label)
        self.is_multiclass_task = is_multiclass(self.solution)

        self.fetch_ingestion_info()
        self.plot_alc = PlotAlc(method="step", transform=None, time_budget=self.time_budget, task_name=self.task_name,
                                area_color='cyan', fill_area=True, model_name=None,
                                clear_figure=True, show_final_score=True,
                                show_title=True, save_path=self.score_dir)
        self.learning_curve = self.get_learning_curve()

    def activate_save_final(self):
        self.save_final = True

    def get_task_name(self, solution_dir):
        """Get the task name from solution directory."""
        solution_names = sorted(ls(os.path.join(solution_dir, "*.solution")))
        if len(solution_names) != 1:  # Assert only one file is found
            logger.warning("{} solution files found: {}! "
                           .format(len(solution_names), solution_names) +
                           "Return `None` as task name.")
            return None
        solution_file = solution_names[0]
        task_name = solution_file.split(os.sep)[-1].split(".")[0]
        return task_name

    def get_ingestion_info(self):
        # TODO 使用共享变量得到 start.txt的内容
        return self.start_info_share_dict

    def fetch_ingestion_info(self):
        """Resolve some information from output of ingestion program. This includes
        especially: `ingestion_start`, `ingestion_pid`, `time_budget`.

        Raises:
          IngestionError if no sign of ingestion starting detected after 1800
          seconds.
        """
        logger.debug("Fetching ingestion info...")

        # Wait 1800 seconds for ingestion to start and write 'start.txt', Otherwise, raise an exception.
        wait_time = 1800
        for i in range(wait_time):
            ingestion_info = self.get_ingestion_info()
            if len(ingestion_info) > 0:
                logger.info("Detected the start of ingestion after {} ".format(i) +
                            "seconds. Start scoring.")
                break
            time.sleep(1)
        else:
            raise IngestionException("[-] Failed: scoring didn't detected the start of " +
                                     "ingestion after {} seconds.".format(wait_time))
        # Get ingestion start time
        ingestion_start = ingestion_info["start_time"]
        # Get ingestion PID
        ingestion_pid = ingestion_info["ingestion_pid"]
        # Get time_budget for ingestion
        assert "time_budget" in ingestion_info
        time_budget = ingestion_info["time_budget"]
        # Set attributes
        self.ingestion_info = ingestion_info
        self.ingestion_start = ingestion_start
        self.ingestion_pid = ingestion_pid
        self.time_budget = time_budget
        logger.debug("Ingestion start time: {}".format(ingestion_start))
        logger.debug("Scoring start time: {}".format(self.start_time))
        logger.debug("Ingestion info successfully fetched.")

    def ingestion_finished(self):
        # if ingestion finish, the size of end_info_share_dict greater than 0
        return len(self.end_info_share_dict) > 0

    def ingestion_is_alive(self):
        return is_process_alive(self.ingestion_pid)

    def kill_ingestion(self):
        terminate_process(self.ingestion_pid)
        assert not self.ingestion_is_alive()

    def get_new_prediction_list(self):
        """
            Fetch new prediction phrase id
        Returns:
            List of new prediction phrase id
        """
        prediction_ids = list(self.prediction_share_dict.keys())
        logger.debug("Prediction ids: {}".format(prediction_ids))
        new_prediction_ids = [pred_id for pred_id in prediction_ids if pred_id not in self.prediction_list_so_far]
        self.new_prediction_list = sorted(new_prediction_ids)
        return self.new_prediction_list

    def compute_score_per_prediction(self):
        """For new predictions found, compute their score using `self.solution`
        and scoring functions in `self.scoring_functions`. Then concatenate
        the list of new predictions to the list of resolved predictions so far.
        """
        for score_name in self.scoring_functions:
            scoring_function = self.scoring_functions[score_name]
            if score_name != "accuracy" or self.is_multiclass_task:
                new_scores = [scoring_function(self.solution, np.array(self.prediction_share_dict[pred_id]))
                              for pred_id in self.new_prediction_list]
                if score_name in self.scores_so_far:
                    self.scores_so_far[score_name] += new_scores
                else:
                    self.scores_so_far[score_name] = new_scores
        # If new predictions are found, update state variables
        if self.new_prediction_list:
            self.prediction_list_so_far += self.new_prediction_list
            num_preds = len(self.prediction_list_so_far)
            self.relative_timestamps = self.get_relative_timestamps()[:num_preds]
            self.learning_curve = self.get_learning_curve()
            self.new_prediction_list = []

    def get_timestamps(self):
        """
        The format in start_info_share_dict likes
            ingestion_pid: 31315
            start_time: 1557269921.7939095
            0: 1557269953.5586617
            1: 1557269956.012751
        Returns:
            start_time, timestamps = 1557269921.7939095, [1557269953.5586617, 1557269956.012751]
        """
        if len(self.start_info_share_dict) > 0:
            start_time = self.start_info_share_dict["start_time"]
            pred_order_num = 0
            timestamps = []
            while pred_order_num in self.start_info_share_dict:
                timestamps.append(self.start_info_share_dict[pred_order_num])
                pred_order_num += 1
            return start_time, timestamps
        else:
            logger.warning("No start_info found in the scoring process. Return `[]` as timestamps.")
            return None, []

    def get_relative_timestamps(self):
        """Get a list of relative timestamps. The beginning has relative timestamp
        zero.
        """
        ingestion_start, timestamps = self.get_timestamps()
        relative_timestamps = [t - ingestion_start for t in timestamps]
        return relative_timestamps

    def write_score(self):
        """Write score and duration to score_dir/scores.txt"""
        score_dir = self.score_dir
        score = self.learning_curve.get_alc()
        duration = self.learning_curve.get_time_used()
        score_filename = os.path.join(score_dir, "scores.txt")
        score_info_dict = {"score": score,  # ALC
                           "Duration": duration,
                           "task_name": self.task_name,
                           "timestamps": self.relative_timestamps,
                           "nauc_scores": self.scores_so_far["nauc"]
                           }
        if self.is_multiclass_task:
            score_info_dict["accuracy"] = self.scores_so_far["accuracy"]
        with open(score_filename, "w") as f:
            f.write("score: " + str(score) + "\n")
            f.write("Duration: " + str(duration) + "\n")
            f.write("timestamps: {}\n".format(self.relative_timestamps))
            f.write("nauc_scores: {}\n".format(self.scores_so_far["nauc"]))
            if self.is_multiclass_task:
                f.write("accuracy: {}\n".format(self.scores_so_far["accuracy"]))
        logger.debug("Wrote to score_filename={} with score={}, duration={}"
                     .format(score_filename, score, duration))
        return score_info_dict

    def get_learning_curve(self, score_name="nauc"):
        timestamps = self.relative_timestamps
        scores = self.scores_so_far[score_name]
        return LearningCurve(timestamps=timestamps, scores=scores,
                             time_budget=self.time_budget,
                             score_name=score_name, task_name=self.task_name,
                             participant_name=self.participant_name,
                             algorithm_name=self.algorithm_name)

    def draw_learning_curve(self, **kwargs):
        """Draw learning curve for one task and save to `score_dir`."""
        self.compute_score_per_prediction()
        scores = self.scores_so_far["nauc"]
        is_multiclass_task = self.is_multiclass_task
        timestamps = self.get_relative_timestamps()
        sorted_pairs = sorted(zip(timestamps, scores))
        start = 0
        time_used = -1
        if len(timestamps) > 0:
            time_used = sorted_pairs[-1][0] - start
            latest_score = sorted_pairs[-1][1]
            if is_multiclass_task:
                accuracy_scores = self.scores_so_far["accuracy"]
                sorted_pairs_acc = sorted(zip(timestamps, accuracy_scores))
                latest_acc = sorted_pairs_acc[-1][1]
        X = [t for t, _ in sorted_pairs]
        Y = [s for _, s in sorted_pairs]
        alc = self.plot_alc.plot_learning_curve(X, Y, save_final=self.save_final)

        return alc, time_used

    def update_score_and_learning_curve(self):
        self.draw_learning_curve()

        # Write score
        score = self.write_score()["score"]
        return score

    def compute_scores_bootstrap(self, scoring_function, solution, prediction, n=10):
        """Compute a list of scores using bootstrap.

           Args:
             scoring function: scoring metric taking y_true and y_pred
             solution: ground truth vector
             prediction: proposed solution
             n: number of scores to compute
        """
        scores = []
        for _ in range(n):  # number of scoring
            size = solution.shape[0]
            idx = np.random.randint(0, size, size)  # bootstrap index
            scores.append(scoring_function(solution[idx], prediction[idx]))
        return scores

    def compute_error_bars(self, n=10):
        """Compute error bars on evaluation with bootstrap.

        Args:
            n: number of times to compute the score (more means more precision)
        Returns:
            (mean, std, var)
        """
        try:
            scoring_function = self.scoring_functions['nauc']
            solution = self.solution
            last_prediction = np.array(self.prediction_share_dict[self.prediction_list_so_far[-1]])
            scores = self.compute_scores_bootstrap(scoring_function, solution, last_prediction, n=n)
            return np.mean(scores), np.std(scores), np.var(scores)
        except Exception as exp:  # not able to compute error bars
            return -1, -1, -1

    def compute_alc_error_bars(self, n=10):
        """ Return mean, std and variance of ALC score with n runs.
            n curves are created:
                For each timestamp, the value of AUC is computed from boostraps of y_true and y_pred.
                During one curve building, we keep the same boostrap index for each prediction timestamp.

            Args:
                n: number of times to compute the score (more means more precision)
            Returns:
                (mean, std, var)
        """
        try:
            scoring_function = self.scoring_functions["nauc"]
            solution = self.solution
            alc_scores = []
            for _ in range(n):  # n learning curves to compute
                scores = []
                size = solution.shape[0]
                idx = np.random.randint(0, size, size)  # bootstrap index
                for pred_id in self.prediction_list_so_far:
                    prediction = np.array(self.prediction_share_dict[pred_id])
                    scores.append(scoring_function(solution[idx], prediction[idx]))
                # create new learning curve
                learning_curve = LearningCurve(timestamps=self.relative_timestamps,  # self.learning_curve.timestamps,
                                               scores=scores,  # list of AUC scores
                                               time_budget=self.time_budget)
                alc_scores.append(learning_curve.get_alc())
            return np.mean(alc_scores), np.std(alc_scores), np.var(alc_scores)
        except Exception as exp:  # not able to compute error bars
            return -1, -1, -1

    def score_new_predictions(self):
        new_prediction_list = self.get_new_prediction_list()
        if len(new_prediction_list) > 0 or self.save_final:
            score = self.update_score_and_learning_curve()
            logger.info("[+] New prediction found. Now number of predictions made = {}"
                        .format(len(self.prediction_list_so_far)))
            logger.info("Current area under learning curve for {}: {:.4f}".format(self.task_name, score))
            if len(self.scores_so_far["nauc"]) > 0:
                logger.info("(2 * AUC - 1) of the latest prediction is {:.4f}."
                            .format(self.scores_so_far["nauc"][-1]))
                if self.is_multiclass_task:
                    logger.info("Accuracy of the latest prediction is {:.4f}."
                                .format(self.scores_so_far["accuracy"][-1]))
