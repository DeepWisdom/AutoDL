#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc    : learning curve for alc

import os
import numpy as np
from sklearn.metrics import auc
import matplotlib.pyplot as plt

from autodl.utils.plot_base_metric import PlotBaseMetric
from autodl.utils.util import transform_time, get_fig_name
from autodl.metrics.scores import auc_step


class LearningCurve(object):

    """Learning curve object for AutoDL challenges. Contains at least an
    increasing list of float as timestamps and another list of the same length
    of the corresponding score at each timestamp.
    """

    def __init__(self, timestamps=None, scores=None, time_budget=1200,
                 score_name=None, task_name=None,
                 participant_name=None, algorithm_name=None, subset='test'):
        """
        Args:
          timestamps: list of float, should be increasing
          scores: list of float, should have the same length as `timestamps`
          time_budget: float, the time budget (for ingestion) of the task
          score_name: string, can be 'nauc' or 'accuracy' (if is multiclass task)
          task_name: string, name of the task, optional
          participant_name: string, name of the participant, optional
          algorithm_name: string, name of the algorithm, optional
        """
        self.timestamps = timestamps or []  # relative timestamps
        self.scores = scores or []
        if len(self.timestamps) != len(self.scores):
            raise ValueError("The number of timestamps should be equal to " +
                             "the number of scores, but got " +
                             "{} and {}".format(len(self.timestamps),
                                                len(self.scores)))
        self.time_budget = time_budget
        self.score_name = score_name or 'nauc'
        self.task_name = task_name
        self.participant_name = participant_name
        self.algorithm_name = algorithm_name

    def __repr__(self):
        return "Learning curve for: participant={}, task={}" \
            .format(self.participant_name, self.task_name)

    def __add__(self, lc):
        if not isinstance(lc, LearningCurve):
            raise ValueError("Can only add two learning curves but got {}.".format(type(lc)))
        if self.time_budget != lc.time_budget:
            raise ValueError("Cannot add two learning curves of different " +
                             "time budget: {} and {}!".format(self.time_budget, lc.time_budget))
        else:
            time_budget = self.time_budget
        if self.score_name != lc.score_name:
            raise ValueError("Cannot add two learning curves of different " +
                             "score names: {} and {}!".format(self.score_name, lc.score_name))
        else:
            score_name = self.score_name
        task_name = self.task_name if self.task_name == lc.task_name else None
        participant_name = self.participant_name \
            if self.participant_name == lc.participant_name else None
        algorithm_name = self.algorithm_name \
            if self.algorithm_name == lc.algorithm_name else None
        # Begin merging scores and timestamps
        new_timestamps = []
        new_scores = []
        # Indices of next point to add
        i = 0
        j = 0
        while i < len(self.timestamps) or j < len(lc.timestamps):
            # When two timestamps are close, append only one timestamp
            if i < len(self.timestamps) and j < len(lc.timestamps) and \
                    np.isclose(self.timestamps[i], lc.timestamps[j]):
                new_timestamps.append(self.timestamps[i])
                new_scores.append(self.scores[i] + lc.scores[j])
                i += 1
                j += 1
                continue
            # In some cases, use the timestamp/score of this learning curve
            if j == len(lc.timestamps) or \
                    (i < len(self.timestamps) and self.timestamps[i] < lc.timestamps[j]):
                new_timestamps.append(self.timestamps[i])
                other_score = 0 if j == 0 else lc.scores[j - 1]
                new_scores.append(self.scores[i] + other_score)
                i += 1
            # In other cases, use the timestamp/score of the other learning curve
            else:
                new_timestamps.append(lc.timestamps[j])
                this_score = 0 if i == 0 else self.scores[i - 1]
                new_scores.append(this_score + lc.scores[j])
                j += 1
        new_lc = LearningCurve(timestamps=new_timestamps,
                               scores=new_scores,
                               time_budget=time_budget,
                               score_name=score_name,
                               task_name=task_name,
                               participant_name=participant_name,
                               algorithm_name=algorithm_name)
        return new_lc

    def __mul__(self, real_number):
        if isinstance(real_number, int):
            real_number = float(real_number)
        if not isinstance(real_number, float):
            raise ValueError("Can only multiply a learning curve by a float but got" +
                             " {}.".format(type(real_number)))
        new_scores = [real_number * s for s in self.scores]
        new_lc = LearningCurve(timestamps=self.timestamps,
                               scores=new_scores,
                               time_budget=self.time_budget,
                               score_name=self.score_name,
                               task_name=self.task_name,
                               participant_name=self.participant_name,
                               algorithm_name=self.algorithm_name)
        return new_lc

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, real_number):
        return self * (1 / real_number)

    def plot(self, method='step', transform=None,
             area_color='cyan', fill_area=True, model_name=None,
             fig=None, show_final_score=True, **kwargs):
        """Plot the learning curve using `matplotlib.pyplot`.

        method: string, can be 'step' or 'trapez'. Decides which drawstyle to use.
            Also effects ALC (Area under Learning Curve)
        transform: callable, for transforming time axis to [0,1] interval, mostly
            optional
        area_color: string or color code, decides the color of the area under curve,
            optional
        fill_area: boolean, whether fill the area under curve with color or not
        model_name: string, if not `None`, will be shown on the legend
        fig: matplotlib.figure.Figure, the figure to plot on. If `None` create a new
            one
        show_final_score: boolean, whether show final score on the figure. Useful
            when overlapping curves
        kwargs: Line2D properties, will be passed for plotting the curve
            see https://matplotlib.org/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D
        """
        timestamps = self.timestamps
        scores = self.scores
        time_budget = self.time_budget
        task_name = self.task_name
        alc, fig = None, None
        # TODO delete
        # alc, fig = plot_learning_curve(timestamps, scores,
        #                                start_time=0, time_budget=time_budget, method=method,
        #                                transform=transform, task_name=task_name,
        #                                area_color=area_color,
        #                                fill_area=fill_area, model_name=model_name,
        #                                clear_figure=False, fig=fig,
        #                                show_final_score=show_final_score, **kwargs)
        return alc, fig

    def get_alc(self, t0=60, method='step'):
        X = [transform_time(t, T=self.time_budget, t0=t0)
             for t in self.timestamps]
        Y = list(self.scores.copy())
        X.insert(0, 0)
        Y.insert(0, 0)
        X.append(1)
        Y.append(Y[-1])
        if method == 'step':
            auc_func = auc_step
        elif method == 'trapez':
            auc_func = auc
        alc = auc_func(X, Y)
        return alc

    def get_time_used(self):
        if len(self.timestamps) > 0:
            return self.timestamps[-1]
        else:
            return 0

    def get_final_score(self):
        if len(self.scores) > 0:
            return self.scores[-1]
        else:
            return 0

    def save_figure(self, output_dir):
        alc, ax = self.plot()
        fig_name = get_fig_name(self.task_name)
        path_to_fig = os.path.join(output_dir, fig_name)
        plt.savefig(path_to_fig)
        plt.close()
