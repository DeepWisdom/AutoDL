################################################################################
# Name:         Ingestion Program
# Author:       Zhengying Liu, Isabelle Guyon, Adrien Pavao, Zhen Xu
# Update time:  13 Aug 2019
# Usage: python ingestion.py --dataset_dir=<dataset_dir> --output_dir=<prediction_dir> --ingestion_program_dir=<ingestion_program_dir> --code_dir=<code_dir> --score_dir=<score_dir>

# AS A PARTICIPANT, DO NOT MODIFY THIS CODE.

VERSION = 'v20191204'
DESCRIPTION =\
"""This is the "ingestion program" written by the organizers. It takes the
code written by participants (with `model.py`) and one dataset as input,
run the code on the dataset and produce predictions on test set. For more
information on the code/directory structure, please see comments in this
code (ingestion.py) and the README file of the starting kit.
Previous updates:
20191204: [ZY] Add timer and separate model initialization from train/predict
               process, : now model initilization doesn't consume time budget
               quota (but can only use 20min)
20190820: [ZY] Mark the beginning of ingestion right before model.py to reduce
               variance
20190708: [ZY] Integrate Julien's parallel data loader
20190516: [ZY] Change time budget to 20 minutes.
20190508: [ZY] Add time_budget to 'start.txt'
20190507: [ZY] Write timestamps to 'start.txt'
20190505: [ZY] Use argparse to parse directories AND time budget;
               Rename input_dir to dataset_dir;
               Rename submission_dir to code_dir;
20190504: [ZY] Check if model.py has attribute done_training and use it to
               determinate whether ingestion has ended;
               Use module-specific logger instead of logging (with root logger);
               At beginning, write start.txt with ingestion_pid and start_time;
               In the end, write end.txt with end_time and ingestion_success;
20190429: [ZY] Remove useless code block; better code layout.
20190425: [ZY] Check prediction shape.
20190424: [ZY] Use logging instead of logger; remove start.txt checking;
20190419: [ZY] Try-except clause for training process;
          always terminates successfully.
"""
# The dataset directory dataset_dir (e.g. AutoDL_sample_data/) contains one dataset
# folder (e.g. adult.data/) with the training set (train/)  and test set (test/),
# each containing an some tfrecords data with a `metadata.textproto` file of
# metadata on the dataset. So one AutoDL dataset will look like
#
#   adult.data
#   ├── test
#   │   ├── metadata.textproto
#   │   └── sample-adult-test.tfrecord
#   └── train
#       ├── metadata.textproto
#       └── sample-adult-train.tfrecord
#
# The output directory output_dir (e.g. AutoDL_sample_result_submission/)
# will first have a start.txt file written by ingestion then receive
# all predictions made during the whole train/predict process
# (thus this directory is updated when a new prediction is made):
# 	adult.predict_0
# 	adult.predict_1
# 	adult.predict_2
#        ...
# after ingestion has finished, a file end.txt will be written, containing
# info on the duration ingestion used. This file is also used as a signal
# for scoring program showing that ingestion has terminated.
#
# The code directory submission_program_dir (e.g. AutoDL_sample_code_submission/)
# should contain your code submission model.py (and possibly other functions
# it depends upon).
#
# We implemented several classes:
# 1) DATA LOADING:
#    ------------
# dataset.py
# dataset.AutoDLMetadata: Read metadata in metadata.textproto
# dataset.AutoDLDataset: Read data and give tf.data.Dataset
# 2) LEARNING MACHINE:
#    ----------------
# model.py
# model.Model.train
# model.Model.test
#
# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
# UNIVERSITE PARIS SUD, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS.
# IN NO EVENT SHALL UNIVERSITE PARIS SUD AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL,
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS,
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE.
#
# Main contributors: Isabelle Guyon and Zhengying Liu

# =========================== BEGIN OPTIONS ==============================

# Verbosity level of logging:
##############
# Can be: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
verbosity_level = 'INFO'

# Some common useful packages
from contextlib import contextmanager
from os import getcwd as pwd
from os.path import join
from sys import argv, path
import argparse
import datetime
import glob
import logging
import math
import numpy as np
import os
import sys
import signal
import time

def get_logger(verbosity_level, use_error_log=False):
  """Set logging format to something like:
       2019-04-25 12:52:51,924 INFO score.py: <message>
  """
  logger = logging.getLogger(__file__)
  logging_level = getattr(logging, verbosity_level)
  logger.setLevel(logging_level)
  formatter = logging.Formatter(
    fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
  stdout_handler = logging.StreamHandler(sys.stdout)
  stdout_handler.setLevel(logging_level)
  stdout_handler.setFormatter(formatter)
  logger.addHandler(stdout_handler)
  if use_error_log:
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)
  logger.propagate = False
  return logger

logger = get_logger(verbosity_level)

def _HERE(*args):
  """Helper function for getting the current directory of this script."""
  h = os.path.dirname(os.path.realpath(__file__))
  return os.path.abspath(os.path.join(h, *args))

def write_start_file(output_dir, start_time=None, time_budget=None,
                     task_name=None):
  """Create start file 'start.txt' in `output_dir` with ingestion's pid and
  start time.

  The content of this file will be similar to:
      ingestion_pid: 1
      task_name: beatriz
      time_budget: 7200
      start_time: 1557923830.3012087
      0: 1557923854.504741
      1: 1557923860.091236
      2: 1557923865.9630117
      3: 1557923872.3627956
      <more timestamps of predictions>
  """
  ingestion_pid = os.getpid()
  start_filename =  'start.txt'
  start_filepath = os.path.join(output_dir, start_filename)
  with open(start_filepath, 'w') as f:
    f.write('ingestion_pid: {}\n'.format(ingestion_pid))
    f.write('task_name: {}\n'.format(task_name))
    f.write('time_budget: {}\n'.format(time_budget))
    f.write('start_time: {}\n'.format(start_time))
  logger.debug("Finished writing 'start.txt' file.")

def write_timestamp(output_dir, predict_idx, timestamp):
  start_filename = 'start.txt'
  start_filepath = os.path.join(output_dir, start_filename)
  with open(start_filepath, 'a') as f:
    f.write('{}: {}\n'.format(predict_idx, timestamp))
  logger.debug("Wrote timestamp {} to 'start.txt' for predition {}."\
               .format(timestamp, predict_idx))

class ModelApiError(Exception):
  pass

class BadPredictionShapeError(Exception):
  pass

class TimeoutException(Exception):
  pass

class Timer:
  def __init__(self):
    self.duration = 0
    self.total = None
    self.remain = None
    self.exec = None

  def set(self, time_budget):
    self.total = time_budget
    self.remain = time_budget
    self.exec = 0

  @contextmanager
  def time_limit(self, pname):
    def signal_handler(signum, frame):
      raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(int(math.ceil(self.remain)))
    start_time = time.time()

    try:
      yield
    finally:
      exec_time = time.time() - start_time
      signal.alarm(0)
      self.exec += exec_time
      self.duration += exec_time
      self.remain = self.total - self.exec

      logger.info("{} success, time spent so far {} sec"\
                  .format(pname, self.exec))

      if self.remain <= 0:
        raise TimeoutException("Timed out for the process: {}!".format(pname))

# =========================== BEGIN PROGRAM ================================

if __name__=="__main__":

    #### Check whether everything went well
    ingestion_success = True

    # Parse directories from input arguments
    root_dir = _HERE(os.pardir)
    default_dataset_dir = join(root_dir, "AutoDL_sample_data")
    default_output_dir = join(root_dir, "AutoDL_sample_result_submission")
    default_ingestion_program_dir = join(root_dir, "AutoDL_ingestion_program")
    default_code_dir = join(root_dir, "AutoDL_sample_code_submission")
    default_score_dir = join(root_dir, "AutoDL_scoring_output")
    default_time_budget = 1200
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str,
                        default=default_dataset_dir,
                        help="Directory storing the dataset (containing " +
                             "e.g. adult.data/)")
    parser.add_argument('--output_dir', type=str,
                        default=default_output_dir,
                        help="Directory storing the predictions. It will " +
                             "contain e.g. [start.txt, adult.predict_0, " +
                             "adult.predict_1, ..., end.txt] when ingestion " +
                             "terminates.")
    parser.add_argument('--ingestion_program_dir', type=str,
                        default=default_ingestion_program_dir,
                        help="Directory storing the ingestion program " +
                             "`ingestion.py` and other necessary packages.")
    parser.add_argument('--code_dir', type=str,
                        default=default_code_dir,
                        help="Directory storing the submission code " +
                             "`model.py` and other necessary packages.")
    parser.add_argument('--score_dir', type=str,
                        default=default_score_dir,
                        help="Directory storing the scoring output " +
                             "e.g. `scores.txt` and `detailed_results.html`.")
    parser.add_argument('--time_budget', type=float,
                        default=default_time_budget,
                        help="Time budget for running ingestion program.")
    args = parser.parse_args()
    logger.debug("Parsed args are: " + str(args))
    logger.debug("-" * 50)
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    ingestion_program_dir= args.ingestion_program_dir
    code_dir= args.code_dir
    score_dir = args.score_dir
    time_budget = args.time_budget
    if dataset_dir.endswith('run/input') and\
       code_dir.endswith('run/program'):
      logger.debug("Since dataset_dir ends with 'run/input' and code_dir "
                  "ends with 'run/program', suppose running on " +
                  "CodaLab platform. Modify dataset_dir to 'run/input_data' "
                  "and code_dir to 'run/submission'. " +
                  "Directory parsing should be more flexible in the code of " +
                  "compute worker: we need explicit directories for " +
                  "dataset_dir and code_dir.")
      dataset_dir = dataset_dir.replace('run/input', 'run/input_data')
      code_dir = code_dir.replace('run/program', 'run/submission')

    # Show directories for debugging
    logger.debug("sys.argv = " + str(sys.argv))
    logger.debug("Using dataset_dir: " + dataset_dir)
    logger.debug("Using output_dir: " + output_dir)
    logger.debug("Using ingestion_program_dir: " + ingestion_program_dir)
    logger.debug("Using code_dir: " + code_dir)

	  # Our libraries
    path.append(ingestion_program_dir)
    path.append(code_dir)
    #IG: to allow submitting the starting kit as sample submission
    path.append(code_dir + '/AutoDL_sample_code_submission')
    import data_io
    from dataset import AutoDLDataset # THE class of AutoDL datasets

    data_io.mkdir(output_dir)

    #### INVENTORY DATA (and sort dataset names alphabetically)
    datanames = data_io.inventory_data(dataset_dir)
    #### Delete zip files and metadata file
    datanames = [x for x in datanames if x.endswith('.data')]

    if len(datanames) != 1:
      raise ValueError("{} datasets found in dataset_dir={}!\n"\
                       .format(len(datanames), dataset_dir) +
                       "Please put only ONE dataset under dataset_dir.")

    basename = datanames[0]

    logger.info("************************************************")
    logger.info("******** Processing dataset " + basename[:-5].capitalize() +
                 " ********")
    logger.info("************************************************")
    logger.debug("Version: {}. Description: {}".format(VERSION, DESCRIPTION))

    ##### Begin creating training set and test set #####
    logger.info("Reading training set and test set...")
    D_train = AutoDLDataset(os.path.join(dataset_dir, basename, "train"))
    D_test = AutoDLDataset(os.path.join(dataset_dir, basename, "test"))
    ##### End creating training set and test set #####

    ## Get correct prediction shape
    num_examples_test = D_test.get_metadata().size()
    output_dim = D_test.get_metadata().get_output_size()
    correct_prediction_shape = (num_examples_test, output_dim)

    # 20 min for participants to initializing and install other packages
    try:
      init_time_budget = 20 * 60 # time budget for initilization.
      timer = Timer()
      timer.set(init_time_budget)
      with timer.time_limit("Initialization"):
        ##### Begin creating model #####
        logger.info("Creating model...this process should not exceed 20min.")
        from model import Model # in participants' model.py
        M = Model(D_train.get_metadata()) # The metadata of D_train and D_test only differ in sample_count
        ###### End creating model ######
    except TimeoutException as e:
      logger.info("[-] Initialization phase exceeded time budget. Move to train/predict phase")
    except Exception as e:
      logger.error("Failed to initializing model.")
      logger.error("Encountered exception:\n" + str(e), exc_info=True)

    # Mark starting time of ingestion
    start = time.time()
    logger.info("="*5 + " Start core part of ingestion program. " +
                "Version: {} ".format(VERSION) + "="*5)

    write_start_file(output_dir, start_time=start, time_budget=time_budget,
                     task_name=basename.split('.')[0])

    try:
      # Check if the model has methods `train` and `test`.
      for attr in ['train', 'test']:
        if not hasattr(M, attr):
          raise ModelApiError("Your model object doesn't have the method " +
                              "`{}`. Please implement it in model.py.")

      # Check if model.py uses new done_training API instead of marking
      # stopping by returning None
      use_done_training_api = hasattr(M, 'done_training')
      if not use_done_training_api:
        logger.warning("Your model object doesn't have an attribute " +
                       "`done_training`. But this is necessary for ingestion " +
                       "program to know whether the model has done training " +
                       "and to decide whether to proceed more training. " +
                       "Please add this attribute to your model.")

      # Keeping track of how many predictions are made
      prediction_order_number = 0

      # Start the CORE PART: train/predict process
      while(not (use_done_training_api and M.done_training)):
        remaining_time_budget = start + time_budget - time.time()
        # Train the model
        logger.info("Begin training the model...")
        M.train(D_train.get_dataset(),
                remaining_time_budget=remaining_time_budget)
        logger.info("Finished training the model.")
        remaining_time_budget = start + time_budget - time.time()
        # Make predictions using the trained model
        logger.info("Begin testing the model by making predictions " +
                     "on test set...")
        Y_pred = M.test(D_test.get_dataset(),
                        remaining_time_budget=remaining_time_budget)
        logger.info("Finished making predictions.")
        if Y_pred is None: # Stop train/predict process if Y_pred is None
          logger.info("The method model.test returned `None`. " +
                      "Stop train/predict process.")
          break
        else: # Check if the prediction has good shape
          prediction_shape = tuple(Y_pred.shape)
          if prediction_shape != correct_prediction_shape:
            raise BadPredictionShapeError(
              "Bad prediction shape! Expected {} but got {}."\
              .format(correct_prediction_shape, prediction_shape)
            )
        # Write timestamp to 'start.txt'
        write_timestamp(output_dir, predict_idx=prediction_order_number,
                        timestamp=time.time())
        # Prediction files: adult.predict_0, adult.predict_1, ...
        filename_test = basename[:-5] + '.predict_' +\
          str(prediction_order_number)
        # Write predictions to output_dir
        data_io.write(os.path.join(output_dir,filename_test), Y_pred)
        prediction_order_number += 1
        logger.info("[+] {0:d} predictions made, time spent so far {1:.2f} sec"\
                     .format(prediction_order_number, time.time() - start))
        remaining_time_budget = start + time_budget - time.time()
        logger.info( "[+] Time left {0:.2f} sec".format(remaining_time_budget))
        if remaining_time_budget<=0:
          break
    except Exception as e:
      ingestion_success = False
      logger.info("Failed to run ingestion.")
      logger.error("Encountered exception:\n" + str(e), exc_info=True)

    # Finishing ingestion program
    end_time = time.time()
    overall_time_spent = end_time - start

    # Write overall_time_spent to a end.txt file
    end_filename =  'end.txt'
    with open(os.path.join(output_dir, end_filename), 'w') as f:
      f.write('ingestion_duration: ' + str(overall_time_spent) + '\n')
      f.write('ingestion_success: ' + str(int(ingestion_success)) + '\n')
      f.write('end_time: ' + str(end_time) + '\n')
      logger.info("Wrote the file {} marking the end of ingestion."\
                  .format(end_filename))
      if ingestion_success:
          logger.info("[+] Done. Ingestion program successfully terminated.")
          logger.info("[+] Overall time spent %5.2f sec " % overall_time_spent)
      else:
          logger.info("[-] Done, but encountered some errors during ingestion.")
          logger.info("[-] Overall time spent %5.2f sec " % overall_time_spent)

    # Copy all files in output_dir to score_dir
    os.system("cp -R {} {}".format(os.path.join(output_dir, '*'), score_dir))
    logger.debug("Copied all ingestion output to scoring output directory.")

    logger.info("[Ingestion terminated]")
