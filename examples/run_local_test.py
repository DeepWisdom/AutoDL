#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc    : the entry of competition local test

import os
import argparse
from multiprocessing import Process, Manager

from autodl.utils.logger import logger
from autodl.auto_ingestion import data_io
from autodl.utils.ingrestion_process import run_ingestion
from autodl.utils.scoring_process import run_scoring


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
