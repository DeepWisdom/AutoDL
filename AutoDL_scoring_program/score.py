################################################################################
# Name:         Scoring Program
# Author:       Zhengying Liu, Isabelle Guyon, Adrien Pavao, Zhen Xu
# Update time:  13 Aug 2019
# Usage: 		python score.py --solution_dir=<solution_dir> --prediction_dir=<prediction_dir> --score_dir=<score_dir>
#           solution_dir contains  e.g. adult.solution
#           prediction_dir should contain e.g. start.txt, adult.predict_0, adult.predict_1,..., end.txt.
#           score_dir should contain scores.txt, detailed_results.html

VERSION = 'v20191204'
DESCRIPTION =\
"""This is the scoring program for AutoDL challenge. It takes the predictions
made by ingestion program as input and compare to the solution file and produce
a learning curve.
Previous updates:
20191204: [ZY] Set scoring waiting time to 30min (1800s)
20190908: [ZY] Add algebraic operations of learning curves
20190823: [ZY] Fix the ALC in learning curve: use auc_step instead of auc
20190820: [ZY] Minor fix: change wait time (for ingestion) from 30s to 90s
20190709: [ZY] Resolve all issues; rearrange some logging messages;
               simplify main function; fix exit_code of run_local_test.py;
20190708: [ZY] Write the class Evaluator for object-oriented scoring program
20190519: [ZY] Use the correct function for computing AUC of step functions
20190516: [ZY] Change time budget to 20 minutes.
20190508: [ZY] Decompose drawing learning curve functions;
               Remove redirect output feature;
               Use AUC instead of BAC;
               Ignore columns with only one class when computing AUC;
               Use step function instead of trapezoidal rule;
               Change t0=300 from t0=1 in time transformation:
                 log(1 + t/t0) / log(1 + T/t0)
               Add second x-axis for the real time in seconds
20190505: [ZY] Use argparse to parse directories AND time budget;
               Fix num_preds not updated error.
20190504: [ZY] Don't raise Exception anymore (for most cases) in order to
               always have 'Finished' for each submission;
               Kill ingestion when time limit is exceeded;
               Use the last modified time of the file 'start.txt' written by
               ingestion as the start time (`ingestion_start`);
               Use module-specific logger instead of logging (with root logger);
               Use the function update_score_and_learning_curve;
20190429: [ZY] Remove useless code block such as the function is_started;
               Better code layout.
20190426.4: [ZY] Fix yaml format in scores.txt (add missing spaces)
20190426.3: [ZY] Use f.write instead of yaml.dump to write scores.txt
20190426.2: [ZY] Add logging info when writing scores and learning curves.
20190426: [ZY] Now write to scores.txt whenever a new prediction is made. This
               way, participants can still get a score when they exceed time
               limit (but the submission's status will be marked as 'Failed').
20190425: [ZY] Add ScoringError and IngestionError: throw error in these cases.
               Participants will get 'Failed' for their error. But a score will
               still by computed if applicable.
               Improve importing order.
               Log CPU usage.
20190424: [ZY] Use logging instead of logger; remove start.txt checking.
20190424: [ZY] Add version and description.
20190419: [ZY] Judge if ingestion is alive by duration.txt; use logger."""

# Scoring program for the AutoDL challenge
# Isabelle Guyon and Zhengying Liu, ChaLearn, April 2018-

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRINGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS.
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL,
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS,
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE.
################################################################################


################################################################################
# User defined constants
################################################################################

# Verbosity level of logging.
# Can be: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
verbosity_level = 'INFO'

from libscores import read_array, sp, ls, mvmean, tiedrank, _HERE, get_logger
from os.path import join
from sys import argv
from sklearn.metrics import auc
import argparse
import base64
import datetime
import matplotlib; matplotlib.use('Agg') # Solve the Tkinter display issue
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
import sys
import time
import yaml
from random import randrange

logger = get_logger(verbosity_level)

################################################################################
# Functions
################################################################################

# Metric used to compute the score of a point on the learning curve
def autodl_auc(solution, prediction, valid_columns_only=True):
  """Compute normarlized Area under ROC curve (AUC).
  Return Gini index = 2*AUC-1 for  binary classification problems.
  Should work for a vector of binary 0/1 (or -1/1)"solution" and any discriminant values
  for the predictions. If solution and prediction are not vectors, the AUC
  of the columns of the matrices are computed and averaged (with no weight).
  The same for all classification problems (in fact it treats well only the
  binary and multilabel classification problems). When `valid_columns` is not
  `None`, only use a subset of columns for computing the score.
  """
  if valid_columns_only:
    valid_columns = get_valid_columns(solution)
    if len(valid_columns) < solution.shape[-1]:
      logger.info("Some columns in solution have only one class, " +
                     "ignoring these columns for evaluation.")
    solution = solution[:, valid_columns].copy()
    prediction = prediction[:, valid_columns].copy()
  label_num = solution.shape[1]
  auc = np.empty(label_num)
  for k in range(label_num):
    r_ = tiedrank(prediction[:, k])
    s_ = solution[:, k]
    if sum(s_) == 0: print("WARNING: no positive class example in class {}"\
                           .format(k + 1))
    npos = sum(s_ == 1)
    nneg = sum(s_ < 1)
    auc[k] = (sum(r_[s_ == 1]) - npos * (npos + 1) / 2) / (nneg * npos)
  return 2 * mvmean(auc) - 1

def accuracy(solution, prediction):
  """Get accuracy of 'prediction' w.r.t true labels 'solution'."""
  epsilon = 1e-15
  # normalize prediction
  prediction_normalized =\
    prediction / (np.sum(np.abs(prediction), axis=1, keepdims=True) + epsilon)
  return np.sum(solution * prediction_normalized) / solution.shape[0]

scoring_functions = {'nauc': autodl_auc,
                     'accuracy': accuracy
                     }

def get_valid_columns(solution):
  """Get a list of column indices for which the column has more than one class.
  This is necessary when computing BAC or AUC which involves true positive and
  true negative in the denominator. When some class is missing, these scores
  don't make sense (or you have to add an epsilon to remedy the situation).

  Args:
    solution: array, a matrix of binary entries, of shape
      (num_examples, num_features)
  Returns:
    valid_columns: a list of indices for which the column has more than one
      class.
  """
  num_examples = solution.shape[0]
  col_sum = np.sum(solution, axis=0)
  valid_columns = np.where(1 - np.isclose(col_sum, 0) -
                               np.isclose(col_sum, num_examples))[0]
  return valid_columns

def is_one_hot_vector(x, axis=None, keepdims=False):
  """Check if a vector 'x' is one-hot (i.e. one entry is 1 and others 0)."""
  norm_1 = np.linalg.norm(x, ord=1, axis=axis, keepdims=keepdims)
  norm_inf = np.linalg.norm(x, ord=np.inf, axis=axis, keepdims=keepdims)
  return np.logical_and(norm_1 == 1, norm_inf == 1)

def is_multiclass(solution):
  """Return if a task is a multi-class classification task, i.e.  each example
  only has one label and thus each binary vector in `solution` only has
  one '1' and all the rest components are '0'.

  This function is useful when we want to compute metrics (e.g. accuracy) that
  are only applicable for multi-class task (and not for multi-label task).

  Args:
    solution: a numpy.ndarray object of shape [num_examples, num_classes].
  """
  return all(is_one_hot_vector(solution, axis=1))

def get_fig_name(task_name):
  """Helper function for getting learning curve figure name."""
  fig_name = "learning-curve-" + task_name + ".png"
  return fig_name

def get_solution(solution_dir):
  """Get the solution array from solution directory."""
  solution_names = sorted(ls(os.path.join(solution_dir, '*.solution')))
  if len(solution_names) != 1: # Assert only one file is found
    logger.warning("{} solution files found: {}! "\
                   .format(len(solution_names), solution_names) +
                   "Return `None` as solution.")
    return None
  solution_file = solution_names[0]
  solution = read_array(solution_file)
  return solution

def get_task_name(solution_dir):
  """Get the task name from solution directory."""
  solution_names = sorted(ls(os.path.join(solution_dir, '*.solution')))
  if len(solution_names) != 1: # Assert only one file is found
    logger.warning("{} solution files found: {}! "\
                   .format(len(solution_names), solution_names) +
                   "Return `None` as task name.")
    return None
  solution_file = solution_names[0]
  task_name = solution_file.split(os.sep)[-1].split('.')[0]
  return task_name

def transform_time(t, T, t0=None):
  if t0 is None:
    t0 = T
  return np.log(1 + t / t0) / np.log(1 + T / t0)

def auc_step(X, Y):
  """Compute area under curve using step function (in 'post' mode)."""
  if len(X) != len(Y):
    raise ValueError("The length of X and Y should be equal but got " +
                     "{} and {} !".format(len(X), len(Y)))
  area = 0
  for i in range(len(X) - 1):
    delta_X = X[i + 1] - X[i]
    area += delta_X * Y[i]
  return area

def plot_learning_curve(timestamps, scores,
                        start_time=0, time_budget=7200, method='step',
                        transform=None, task_name=None,
                        area_color='cyan', fill_area=True, model_name=None,
                        clear_figure=True, fig=None, show_final_score=True,
                        show_title=True,
                        **kwargs):
  """Plot learning curve using scores and corresponding timestamps.

  Args:
    timestamps: iterable of float, each element is the timestamp of
      corresponding performance. These timestamps should be INCREASING.
    scores: iterable of float, scores at each timestamp
    start_time: float, the start time, should be smaller than any timestamp
    time_budget: float, the time budget, should be larger than any timestamp
    method: string, can be one of ['step', 'trapez']
    transform: callable that transform [0, time_budget] into [0, 1]. If `None`,
      use the default transformation
          lambda t: np.log2(1 + t / time_budget)
    task_name: string, name of the task
    area_color: matplotlib color, color of the area under learning curve
    fill_area: boolean, fill the area under the curve or not
    model_name: string, name of the model (learning algorithm).
    clear_figure: boolean, clear previous figures or not
    fig: the figure to plot on
    show_final_score: boolean, show the last score or not
    show_title: boolean, show the plot title or not
    kwargs: Line2D properties, optional
  Returns:
    alc: float, the area under learning curve.
    fig: the figure with learning curve
  Raises:
    ValueError: if the length of `timestamps` and `scores` are not equal,
      or if `timestamps` is not increasing, or if certain timestamp is not in
      the interval [start_time, start_time + time_budget], or if `method` has
      bad values.
  """
  le = len(timestamps)
  if not le == len(scores):
    raise ValueError("The number of timestamps {} ".format(le) +
                     "should be equal to the number of " +
                     "scores {}!".format(len(scores)))
  for i in range(le):
    if i < le - 1 and not timestamps[i] <= timestamps[i + 1]:
      raise ValueError("The timestamps should be increasing! But got " +
                       "[{}, {}] ".format(timestamps[i], timestamps[i + 1]) +
                       "at index [{}, {}].".format(i, i + 1))
    if timestamps[i] < start_time:
      raise ValueError("The timestamp {} at index {}".format(timestamps[i], i) +
                       " is earlier than start time {}!".format(start_time))
  timestamps = [t for t in timestamps if t <= time_budget + start_time]
  if len(timestamps) < le:
    logger.warning("Some predictions are made after the time budget! " +
                   "Ignoring all predictions from the index {}."\
                   .format(len(timestamps)))
    scores = scores[:len(timestamps)]
  if transform is None:
    t0 = 60
    # default transformation
    transform = lambda t: transform_time(t, time_budget, t0=t0)
    xlabel = "Transformed time: " +\
             r'$\tilde{t} = \frac{\log (1 + t / t_0)}{ \log (1 + T / t_0)}$ ' +\
             ' ($T = ' + str(int(time_budget)) + '$, ' +\
             ' $t_0 = ' + str(int(t0)) + '$)'
  else:
    xlabel = "Transformed time: " + r'$\tilde{t}$'
  relative_timestamps = [t - start_time for t in timestamps]
  # Transform X
  X = [transform(t) for t in relative_timestamps]
  Y = list(scores.copy())
  # Add origin as the first point of the curve
  X.insert(0, 0)
  Y.insert(0, 0)
  # Draw learning curve
  if clear_figure:
    plt.clf()
  if fig is None or len(fig.axes) == 0:
    fig = plt.figure(figsize=(7, 7.07))
    ax = fig.add_subplot(111)
    if show_title:
      plt.title("Learning curve for task: {}".format(task_name), y=1.06)
    ax.set_xlabel(xlabel)
    ax.set_xlim(left=0, right=1)
    ax.set_ylabel('score (2 * AUC - 1)')
    ax.set_ylim(bottom=-0.01, top=1)
    ax.grid(True, zorder=5)
    # Show real time in seconds in a second x-axis
    ax2 = ax.twiny()
    ticks = [10, 60, 300, 600, 1200] +\
            list(range(1800, int(time_budget) + 1, 1800))
    ax2.set_xticks([transform(t) for t in ticks])
    ax2.set_xticklabels(ticks)
  ax = fig.axes[0]
  if method == 'step':
    drawstyle = 'steps-post'
    step = 'post'
    auc_func = auc_step
  elif method == 'trapez':
    drawstyle = 'default'
    step = None
    auc_func = auc
  else:
    raise ValueError("The `method` variable should be one of " +
                     "['step', 'trapez']!")
  # Add a point on the final line using last prediction
  X.append(1)
  Y.append(Y[-1])
  # Compute AUC using step function rule or trapezoidal rule
  alc = auc_func(X, Y)
  if model_name:
    label = "{}: ALC={:.4f}".format(model_name, alc)
  else:
    label = "ALC={:.4f}".format(alc)
  # Plot the major part of the figure: the curve
  if 'marker' not in kwargs:
    kwargs['marker'] = 'o'
  if 'markersize' not in kwargs:
    kwargs['markersize'] = 3
  if 'label' not in kwargs:
    kwargs['label'] = label
  ax.plot(X[:-1], Y[:-1], drawstyle=drawstyle, **kwargs)
  # Fill area under the curve
  if fill_area:
    ax.fill_between(X, Y, color='cyan', step=step)
  # Show the latest/final score
  if show_final_score:
    ax.text(X[-1], Y[-1], "{:.4f}".format(Y[-1]))
  # Draw a dotted line from last prediction
  kwargs['linestyle'] = '--'
  kwargs['linewidth'] = 1
  kwargs['marker'] = None
  kwargs.pop('label', None)
  ax.plot(X[-2:], Y[-2:], **kwargs)
  ax.legend()
  return alc, fig

def get_ingestion_info(prediction_dir):
  """Get info on ingestion program: PID, start time, etc. from 'start.txt'.

  Args:
    prediction_dir: a string, directory containing predictions (output of
      ingestion)
  Returns:
    A dictionary with keys 'ingestion_pid' and 'start_time' if the file
      'start.txt' exists. Otherwise return `None`.
  """
  start_filepath = os.path.join(prediction_dir, 'start.txt')
  if os.path.exists(start_filepath):
    with open(start_filepath, 'r') as f:
      ingestion_info = yaml.safe_load(f)
    return ingestion_info
  else:
    return None

def get_timestamps(prediction_dir):
  """Read predictions' timestamps stored in 'start.txt'.

  The 'start.txt' file should be similar to
    ingestion_pid: 31315
    start_time: 1557269921.7939095
    0: 1557269953.5586617
    1: 1557269956.012751
    2: 1557269958.3
  We see there are 3 predictions. Then this function will return
    start_time, timestamps =
      1557269921.7939095, [1557269953.5586617, 1557269956.012751, 1557269958.3]
  """
  start_filepath = os.path.join(prediction_dir, 'start.txt')
  if os.path.exists(start_filepath):
    with open(start_filepath, 'r') as f:
      ingestion_info = yaml.safe_load(f)
    start_time = ingestion_info['start_time']
    timestamps = []
    idx = 0
    while idx in ingestion_info:
      timestamps.append(ingestion_info[idx])
      idx += 1
    return start_time, timestamps
  else:
    logger.warning("No 'start.txt' file found in the prediction directory " +
                   "{}. Return `None` as timestamps.")
    return None

def get_scores(scoring_function, solution, predictions):
  """Compute a list of scores for a list of predictions.

  Args:
    scoring_function: callable with signature
      scoring_function(solution, predictions)
    solution: Numpy array, the solution (true labels).
    predictions: list of array, predictions.
  Returns:
    a list of float, scores
  """
  scores = [scoring_function(solution, pred) for pred in predictions]
  return scores

def compute_scores_bootstrap(scoring_function, solution, prediction, n=10):
    """Compute a list of scores using bootstrap.

       Args:
         scoring function: scoring metric taking y_true and y_pred
         solution: ground truth vector
         prediction: proposed solution
         n: number of scores to compute
    """
    scores = []
    l = len(solution)
    for _ in range(n): # number of scoring
      size = solution.shape[0]
      idx = np.random.randint(0, size, size) # bootstrap index
      scores.append(scoring_function(solution[idx], prediction[idx]))
    return scores

def end_file_generated(prediction_dir):
  """Check if ingestion is still alive by checking if the file 'end.txt'
  is generated in the folder of predictions.
  """
  end_filepath =  os.path.join(prediction_dir, 'end.txt')
  logger.debug("CPU usage: {}%".format(psutil.cpu_percent()))
  logger.debug("Virtual memory: {}".format(psutil.virtual_memory()))
  return os.path.isfile(end_filepath)

def is_process_alive(pid):
  """Check if a process is alive according to its PID."""
  try:
    os.kill(pid, 0)
  except OSError:
    return False
  else:
    return True

def terminate_process(pid):
  """Kill a process according to its PID."""
  process = psutil.Process(pid)
  process.terminate()
  logger.debug("Terminated process with pid={} in scoring.".format(pid))

class IngestionError(Exception):
  pass

class ScoringError(Exception):
  pass

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
    self.timestamps = timestamps or [] # relative timestamps
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
    return "Learning curve for: participant={}, task={}"\
           .format(self.participant_name, self.task_name)

  def __add__(self, lc):
    if not isinstance(lc, LearningCurve):
      raise ValueError("Can only add two learning curves but got {}."\
                       .format(type(lc)))
    if self.time_budget != lc.time_budget:
      raise ValueError("Cannot add two learning curves of different " +
                       "time budget: {} and {}!"\
                       .format(self.time_budget, lc.time_budget))
    else:
      time_budget = self.time_budget
    if self.score_name != lc.score_name:
      raise ValueError("Cannot add two learning curves of different " +
                       "score names: {} and {}!"\
                       .format(self.score_name, lc.score_name))
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
    alc, fig = plot_learning_curve(timestamps, scores,
                  start_time=0, time_budget=time_budget, method=method,
                  transform=transform, task_name=task_name,
                  area_color=area_color,
                  fill_area=fill_area, model_name=model_name,
                  clear_figure=False, fig=fig,
                  show_final_score=show_final_score, **kwargs)
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

class Evaluator(object):

  def __init__(self, solution_dir=None, prediction_dir=None, score_dir=None,
               scoring_functions=None, task_name=None, participant_name=None,
               algorithm_name=None, submission_id=None):
    """
    Args:
      scoring_functions: a dict containing (string, scoring_function) pairs
    """
    self.start_time = time.time()

    self.solution_dir = solution_dir
    self.prediction_dir = prediction_dir
    self.score_dir = score_dir
    self.scoring_functions = scoring_functions

    self.task_name = task_name or get_task_name(solution_dir)
    self.participant_name = participant_name
    self.algorithm_name = algorithm_name
    self.submission_id = submission_id

    # State variables
    self.scoring_success = None
    self.time_limit_exceeded = None
    self.prediction_files_so_far = []
    self.new_prediction_files = []
    self.scores_so_far = {'nauc':[]}
    self.relative_timestamps = []

    # Resolve info from directories
    self.solution = self.get_solution()
    # Check if the task is multilabel (i.e. with one hot label)
    self.is_multiclass_task = is_multiclass(self.solution)

    self.initialize_learning_curve_page()
    self.fetch_ingestion_info()
    self.learning_curve = self.get_learning_curve()

  def get_solution(self):
    """Get solution as NumPy array from `self.solution_dir`."""
    solution = get_solution(self.solution_dir)
    logger.debug("Successfully loaded solution from solution_dir={}"\
                 .format(self.solution_dir))
    return solution

  def initialize_learning_curve_page(self):
    """Initialize learning curve page with a message for waiting."""
    # Create the output directory, if it does not already exist
    if not os.path.isdir(self.score_dir):
      os.mkdir(self.score_dir)
    # Initialize detailed_results.html (learning curve page)
    detailed_results_filepath = os.path.join(self.score_dir,
                                             'detailed_results.html')
    html_head = '<html><head> <meta http-equiv="refresh" content="5"> ' +\
                '</head><body><pre>'
    html_end = '</pre></body></html>'
    with open(detailed_results_filepath, 'a') as html_file:
      html_file.write(html_head)
      html_file.write("Starting training process... <br> Please be patient. " +
                      "Learning curves will be generated when first " +
                      "predictions are made.")
      html_file.write(html_end)

  def fetch_ingestion_info(self):
    """Resolve some information from output of ingestion program. This includes
    especially: `ingestion_start`, `ingestion_pid`, `time_budget`.

    Raises:
      IngestionError if no sign of ingestion starting detected after 1800
      seconds.
    """
    logger.debug("Fetching ingestion info...")
    prediction_dir = self.prediction_dir
    # Wait 1800 seconds for ingestion to start and write 'start.txt',
    # Otherwise, raise an exception.
    wait_time = 1800
    ingestion_info = None
    for i in range(wait_time):
      ingestion_info = get_ingestion_info(prediction_dir)
      if not ingestion_info is None:
        logger.info("Detected the start of ingestion after {} ".format(i) +
                    "seconds. Start scoring.")
        break
      time.sleep(1)
    else:
      raise IngestionError("[-] Failed: scoring didn't detected the start of " +
                           "ingestion after {} seconds.".format(wait_time))
    # Get ingestion start time
    ingestion_start = ingestion_info['start_time']
    # Get ingestion PID
    ingestion_pid = ingestion_info['ingestion_pid']
    # Get time_budget for ingestion
    assert 'time_budget' in ingestion_info
    time_budget = ingestion_info['time_budget']
    # Set attributes
    self.ingestion_info = ingestion_info
    self.ingestion_start = ingestion_start
    self.ingestion_pid = ingestion_pid
    self.time_budget = time_budget
    logger.debug("Ingestion start time: {}".format(ingestion_start))
    logger.debug("Scoring start time: {}".format(self.start_time))
    logger.debug("Ingestion info successfully fetched.")

  def end_file_generated(self):
    return end_file_generated(self.prediction_dir)

  def ingestion_is_alive(self):
    return is_process_alive(self.ingestion_pid)

  def kill_ingestion(self):
    terminate_process(self.ingestion_pid)
    assert not self.ingestion_is_alive()

  def prediction_filename_pattern(self):
    return "{}.predict_*".format(self.task_name)

  def get_new_prediction_files(self):
    """Fetch new prediction file(name)s found in prediction directory and update
    corresponding attributes.

    Examples of prediction file name: mini.predict_0, mini.predict_1

    Returns:
      List of new prediction files found.
    """
    prediction_files = ls(os.path.join(self.prediction_dir,
                                       self.prediction_filename_pattern()))
    logger.debug("Prediction files: {}".format(prediction_files))
    new_prediction_files = [p for p in prediction_files
                            if p not in self.prediction_files_so_far]
    order_key = lambda filename: int(filename.split('_')[-1])
    self.new_prediction_files = sorted(new_prediction_files, key=order_key)
    return self.new_prediction_files

  def compute_score_per_prediction(self):
    """For new predictions found, compute their score using `self.solution`
    and scoring functions in `self.scoring_functions`. Then concatenate
    the list of new predictions to the list of resolved predictions so far.
    """
    for score_name in self.scoring_functions:
      scoring_function = self.scoring_functions[score_name]
      if score_name != 'accuracy' or self.is_multiclass_task:
        new_scores = [scoring_function(self.solution, read_array(pred))
                      for pred in self.new_prediction_files]
        if score_name in self.scores_so_far:
          self.scores_so_far[score_name] += new_scores
        else:
          self.scores_so_far[score_name] = new_scores
    # If new predictions are found, update state variables
    if self.new_prediction_files:
      self.prediction_files_so_far += self.new_prediction_files
      num_preds = len(self.prediction_files_so_far)
      self.relative_timestamps = self.get_relative_timestamps()[:num_preds]
      self.learning_curve = self.get_learning_curve()
      self.new_prediction_files = []

  def get_relative_timestamps(self):
    """Get a list of relative timestamps. The beginning has relative timestamp
    zero.
    """
    ingestion_start, timestamps = get_timestamps(self.prediction_dir)
    relative_timestamps = [t - ingestion_start for t in timestamps]
    return relative_timestamps

  def write_score(self):
    """Write score and duration to score_dir/scores.txt"""
    score_dir = self.score_dir
    score = self.learning_curve.get_alc()
    duration = self.learning_curve.get_time_used()
    score_filename = os.path.join(score_dir, 'scores.txt')
    score_info_dict = {'score': score, # ALC
                       'Duration': duration,
                       'task_name': self.task_name,
                       'timestamps': self.relative_timestamps,
                       'nauc_scores': self.scores_so_far['nauc']
                      }
    if self.is_multiclass_task:
      score_info_dict['accuracy'] = self.scores_so_far['accuracy']
    with open(score_filename, 'w') as f:
      f.write('score: ' + str(score) + '\n')
      f.write('Duration: ' + str(duration) + '\n')
      f.write('timestamps: {}\n'.format(self.relative_timestamps))
      f.write('nauc_scores: {}\n'.format(self.scores_so_far['nauc']))
      if self.is_multiclass_task:
        f.write('accuracy: {}\n'.format(self.scores_so_far['accuracy']))
    logger.debug("Wrote to score_filename={} with score={}, duration={}"\
                  .format(score_filename, score, duration))
    return score_info_dict

  def write_scores_html(self, auto_refresh=True, append=False):
    score_dir = self.score_dir
    filename = 'detailed_results.html'
    image_paths = sorted(ls(os.path.join(score_dir, '*.png')))
    if auto_refresh:
      html_head = '<html><head> <meta http-equiv="refresh" content="5"> ' +\
                  '</head><body><pre>'
    else:
      html_head = """<html><body><pre>"""
    html_end = '</pre></body></html>'
    if append:
      mode = 'a'
    else:
      mode = 'w'
    filepath = os.path.join(score_dir, filename)
    with open(filepath, mode) as html_file:
        html_file.write(html_head)
        for image_path in image_paths:
          with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            encoded_string = encoded_string.decode('utf-8')
            s = '<img src="data:image/png;charset=utf-8;base64,{}"/>'\
                .format(encoded_string)
            html_file.write(s + '<br>')
        html_file.write(html_end)
    logger.debug("Wrote learning curve page to {}".format(filepath))

  def get_learning_curve(self, score_name='nauc'):
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
    scores = self.scores_so_far['nauc']
    is_multiclass_task = self.is_multiclass_task
    timestamps = self.get_relative_timestamps()
    sorted_pairs = sorted(zip(timestamps, scores))
    start = 0
    time_used = -1
    if len(timestamps) > 0:
      time_used = sorted_pairs[-1][0] - start
      latest_score = sorted_pairs[-1][1]
      if is_multiclass_task:
        accuracy_scores = self.scores_so_far['accuracy']
        sorted_pairs_acc = sorted(zip(timestamps, accuracy_scores))
        latest_acc = sorted_pairs_acc[-1][1]
    X = [t for t, _ in sorted_pairs]
    Y = [s for _, s in sorted_pairs]
    alc, fig = plot_learning_curve(X, Y, time_budget=self.time_budget,
                            task_name=self.task_name, **kwargs)
    fig_name = get_fig_name(self.task_name)
    path_to_fig = os.path.join(self.score_dir, fig_name)
    plt.savefig(path_to_fig)
    plt.close()
    return alc, time_used

  def update_score_and_learning_curve(self):
    self.draw_learning_curve()
    # Update learning curve page (detailed_results.html)
    self.write_scores_html()
    # Write score
    score = self.write_score()['score']
    return score

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
        last_prediction = read_array(self.prediction_files_so_far[-1])
        scores = compute_scores_bootstrap(scoring_function, solution, last_prediction, n=n)
        return np.mean(scores), np.std(scores), np.var(scores)
    except: # not able to compute error bars
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
          scoring_function = self.scoring_functions['nauc']
          solution = self.solution
          alc_scores = []
          for _ in range(n): # n learning curves to compute
              scores = []
              size = solution.shape[0]
              idx = np.random.randint(0, size, size) # bootstrap index
              for prediction_file in self.prediction_files_so_far:
                  prediction = read_array(prediction_file)
                  scores.append(scoring_function(solution[idx], prediction[idx]))
              # create new learning curve
              learning_curve = LearningCurve(timestamps=self.relative_timestamps, # self.learning_curve.timestamps,
                                             scores=scores, # list of AUC scores
                                             time_budget=self.time_budget)
              alc_scores.append(learning_curve.get_alc())
          return np.mean(alc_scores), np.std(alc_scores), np.var(alc_scores)
      except: # not able to compute error bars
          return -1, -1, -1

  def score_new_predictions(self):
    new_prediction_files = evaluator.get_new_prediction_files()
    if len(new_prediction_files) > 0:
      score = evaluator.update_score_and_learning_curve()
      logger.info("[+] New prediction found. Now number of predictions " +
                   "made = {}"\
                   .format(len(evaluator.prediction_files_so_far)))
      logger.info("Current area under learning curve for {}: {:.4f}"\
                .format(evaluator.task_name, score))
      logger.info("(2 * AUC - 1) of the latest prediction is {:.4f}."\
                .format(evaluator.scores_so_far['nauc'][-1]))
      if evaluator.is_multiclass_task:
        logger.info("Accuracy of the latest prediction is {:.4f}."\
                  .format(evaluator.scores_so_far['accuracy'][-1]))

# =============================== MAIN ========================================

if __name__ == "__main__":
    logger.info("="*5 + " Start scoring program. " +
                "Version: {} ".format(VERSION) + "="*5)

    # Default I/O directories:
    root_dir = _HERE(os.pardir)
    default_solution_dir = join(root_dir, "AutoDL_sample_data")
    default_prediction_dir = join(root_dir, "AutoDL_sample_result_submission")
    default_score_dir = join(root_dir, "AutoDL_scoring_output")

    # Parse directories from input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--solution_dir', type=str,
                        default=default_solution_dir,
                        help="Directory storing the solution with true " +
                             "labels, e.g. adult.solution.")
    parser.add_argument('--prediction_dir', type=str,
                        default=default_prediction_dir,
                        help="Directory storing the predictions. It should" +
                             "contain e.g. [start.txt, adult.predict_0, " +
                             "adult.predict_1, ..., end.txt].")
    parser.add_argument('--score_dir', type=str,
                        default=default_score_dir,
                        help="Directory storing the scoring output " +
                             "e.g. `scores.txt` and `detailed_results.html`.")
    args = parser.parse_args()
    logger.debug("Parsed args are: " + str(args))
    logger.debug("-" * 50)
    solution_dir = args.solution_dir
    prediction_dir = args.prediction_dir
    score_dir = args.score_dir

    logger.debug("Version: {}. Description: {}".format(VERSION, DESCRIPTION))
    logger.debug("Using solution_dir: " + str(solution_dir))
    logger.debug("Using prediction_dir: " + str(prediction_dir))
    logger.debug("Using score_dir: " + str(score_dir))

    #################################################################
    # Initialize an evaluator (scoring program) object
    evaluator = Evaluator(solution_dir, prediction_dir, score_dir,
                          scoring_functions=scoring_functions)
    #################################################################

    ingestion_start = evaluator.ingestion_start
    time_budget = evaluator.time_budget

    try:
      while(time.time() < ingestion_start + time_budget):
        if evaluator.end_file_generated():
          logger.info("Detected ingestion program had stopped running " +
                      "because an 'end.txt' file is written by ingestion. " +
                      "Stop scoring now.")
          evaluator.scoring_success = True
          break
        time.sleep(1)

        ### Fetch new predictions, compute their scores and update variables ###
        evaluator.score_new_predictions()
        ########################################################################

        logger.debug("Prediction files so far: {}"\
                     .format(evaluator.prediction_files_so_far))
      else: # When time budget is used up, kill ingestion
        if evaluator.ingestion_is_alive():
          evaluator.time_limit_exceeded = True
          evaluator.kill_ingestion()
          logger.info("Detected time budget is used up. Killed ingestion and " +
                      "terminating scoring...")
    except Exception as e:
      evaluator.scoring_success = False
      logger.error("[-] Error occurred in scoring:\n" + str(e),
                    exc_info=True)

    evaluator.score_new_predictions()

    logger.info("Final area under learning curve for {}: {:.4f}"\
              .format(evaluator.task_name, evaluator.learning_curve.get_alc()))

    # Write one last time the detailed results page without auto-refreshing
    evaluator.write_scores_html(auto_refresh=False)

    # Compute scoring error bars of last prediction
    n = 10
    logger.info("Computing error bars with {} scorings...".format(n))
    mean, std, var = evaluator.compute_error_bars(n=n)
    logger.info("\nLatest prediction NAUC:\n* Mean: {}\n* Standard deviation: {}\n* Variance: {}".format(mean, std, var))

    # Compute ALC error bars
    n = 5
    logger.info("Computing ALC error bars with {} curves...".format(n))
    mean, std, var = evaluator.compute_alc_error_bars(n=n)
    logger.info("\nArea under Learning Curve:\n* Mean: {}\n* Standard deviation: {}\n* Variance: {}".format(mean, std, var))

    scoring_start = evaluator.start_time
    # Use 'end.txt' file to detect if ingestion program ends
    end_filepath =  os.path.join(prediction_dir, 'end.txt')
    if not evaluator.scoring_success is None and not evaluator.scoring_success:
      logger.error("[-] Some error occurred in scoring program. " +
                  "Please see output/error log of Scoring Step.")
    elif not os.path.isfile(end_filepath):
      if evaluator.time_limit_exceeded:
        logger.error("[-] Ingestion program exceeded time budget. " +
                     "Predictions made so far will be used for evaluation.")
      else: # Less probable to fall in this case
        if evaluator.ingestion_is_alive():
          evaluator.kill_ingestion()
        logger.error("[-] No 'end.txt' file is produced by ingestion. " +
                     "Ingestion or scoring may have not terminated normally.")
    else:
      with open(end_filepath, 'r') as f:
        end_info_dict = yaml.safe_load(f)
      ingestion_duration = end_info_dict['ingestion_duration']

      if end_info_dict['ingestion_success'] == 0:
        logger.error("[-] Some error occurred in ingestion program. " +
                    "Please see output/error log of Ingestion Step.")
      else:
        logger.info("[+] Successfully finished scoring! " +
                  "Scoring duration: {:.2f} sec. "\
                  .format(time.time() - scoring_start) +
                  "Ingestion duration: {:.2f} sec. "\
                  .format(ingestion_duration) +
                  "The score of your algorithm on the task '{}' is: {:.6f}."\
                  .format(evaluator.task_name,
                          evaluator.learning_curve.get_alc()))

    logger.info("[Scoring terminated]")
