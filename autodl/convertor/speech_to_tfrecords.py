# Convert AutoSpeech datasets into TFRecords format
# Date: 26 Sept 2019

# Usage: `python3 speech_to_tfrecords.py path/to/dataset`

# Input format files tree:
# ├── AutoSpeech Dataset (name)
#     ├── name.solution (test solution)
#     ├── name.data
#         ├── meta.json
#         ├── train.pkl (2D list, each line is one example)
#         ├── test.pkl
#         ├── train.solution

import os
import pickle
from sys import argv

from autodl.convertor.dataset_formatter import UniMediaDatasetFormatter


def read_data(filename):
    """ Open pickle file
        Return a list of ndarrays
    """
    f = open(filename, 'rb')
    return pickle.load(f)


def read_file(filename):
    f = open(filename, 'r')
    output = f.read().split('\n')
    if '' in output:
        output.remove('')
    f.close()
    return output


def get_features(row):
    """
    Args:
      row: ndarray, time series of float numbers
    Returns:
      a list for dense representation
    """
    features = []
    for e in row:
        features.append([e])
    return features


def get_labels(row):
    labels = row.split(' ')
    return list(map(int, labels))


def get_features_labels_pairs(data, solution):
    # Function that returns a generator of pairs (features, labels)
    def func(i):
        features = get_features(data[i])
        labels = get_labels(solution[i])
        return features, labels
    g = iter(range(len(data)))
    features_labels_pairs = lambda:map(func, g)
    return features_labels_pairs


def get_output_dim(solution):
    return len(solution[0].split(' '))


def autospeech_2_autodl_format(input_dir: str):
    input_dir = os.path.normpath(input_dir)
    name = os.path.basename(input_dir)
    output_dir = input_dir + '_formatted'

    # Read data
    train_data = read_data(os.path.join(input_dir, name+'.data', 'train.pkl'))
    train_solution = read_file(os.path.join(input_dir, name+'.data', 'train.solution'))
    test_data = read_data(os.path.join(input_dir, name+'.data', 'test.pkl'))
    test_solution = read_file(os.path.join(input_dir, name+'.solution'))

    # Convert data into sequences of integers
    features_labels_pairs_train = get_features_labels_pairs(train_data, train_solution)
    features_labels_pairs_test = get_features_labels_pairs(test_data, test_solution)

    # Write data in TFRecords format
    output_dim = get_output_dim(train_solution)
    col_count, row_count = 1, 1
    sequence_size = -1
    num_channels = 1
    num_examples_train = len(train_data)
    num_examples_test = len(test_data)
    new_dataset_name = name  # same name
    classes_list = None
    channels_list = None
    dataset_formatter = UniMediaDatasetFormatter(name,
                                                 output_dir,
                                                 features_labels_pairs_train,
                                                 features_labels_pairs_test,
                                                 output_dim,
                                                 col_count,
                                                 row_count,
                                                 sequence_size=sequence_size,  # for strides=2
                                                 num_channels=num_channels,
                                                 num_examples_train=num_examples_train,
                                                 num_examples_test=num_examples_test,
                                                 is_sequence_col='false',
                                                 is_sequence_row='false',
                                                 has_locality_col='true',
                                                 has_locality_row='true',
                                                 format='DENSE',
                                                 label_format='DENSE',
                                                 is_sequence='false',
                                                 sequence_size_func=None,
                                                 new_dataset_name=new_dataset_name,
                                                 classes_list=classes_list,
                                                 channels_list=channels_list)
    dataset_formatter.press_a_button_and_give_me_an_AutoDL_dataset()


if __name__ == "__main__":

    if len(argv) == 2:
        input_dir = argv[1]
    else:
        print('Please enter a dataset directory. Usage: `python3 speech_to_tfrecords path/to/dataset`')
        input_dir = None
        exit()

    autospeech_2_autodl_format(input_dir=input_dir)
