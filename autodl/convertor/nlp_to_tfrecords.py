# Convert AutoNLP datasets into TFRecords format
# Date: 27 Aug 2019

# Usage: `python3 nlp_to_tfrecords.py path/to/dataset`

# Input format files tree:
# ├── AutoNLP Dataset (name)
#     ├── name.solution (test solution)
#     ├── name.data
#         ├── meta.json
#         ├── train.data (each line is a string representing one example)
#         ├── test.data
#         ├── train.solution

import os
from sys import argv, path
import json
from autodl.convertor.dataset_formatter import UniMediaDatasetFormatter


def get_language(filename):
    with open(filename) as json_file:
        info = json.load(json_file)
        language = info['language']
    return language


def read_file(filename):
    f = open(filename, 'r')
    output = f.read().split('\n')
    if '' in output:
        output.remove('')
    f.close()
    return output


def clean_token(token):
    return repr(token)[1:-1]  # repr(repr(token))[2:-2]


def create_vocabulary(data, language='EN'):
    print('Creating vocabulary...')
    vocabulary = {}
    i = 0
    for row in data:
        if language != 'ZH':
            row = row.split(' ')
        for token in row:
            # Split (EN or ZH)
            cleaned_token = clean_token(token)
            if cleaned_token not in vocabulary:
                vocabulary[cleaned_token] = i
                i += 1
    return vocabulary


def get_features(row, vocabulary, language='EN', format='DENSE'):
    """
    Args:
      row: string, a sentence in certain language (e.g. EN or ZH)
      vocabulary: dict, mapping from token to its index
      language: string, can be 'EN' or 'ZH'
    Returns:
      if DENSE format:
        a list of 4-tuples of form (row, col, channel)
      if SPARSE format:
        a list of 4-tuples of form (row_index, col_index, channel_index, value)
        for a sparse representation of a 3-D Tensor.
    """
    features = []
    if language != 'ZH':
        row = row.split(' ')
    for e in row:
        token = clean_token(e)
        # if format=='DENSE':
        #     one_hot_word = [0]*len(vocabulary)
        #     one_hot_word[vocabulary[token]] = 1
        #     features.append(one_hot_word)
        # elif format=='SPARSE':
        #     features.append((0, 0, vocabulary[token], 1))
        features.append([vocabulary[token]])
        # else:
        #     raise Exception('Unknown format: {}'.format(format))
    return features


def get_labels(row):
    labels = row.split(' ')
    return list(map(int, labels))


def get_features_labels_pairs(data, solution, vocabulary, language, format='DENSE'):
    # Function that returns a generator of pairs (features, labels)
    def func(i):
        features = get_features(data[i], vocabulary, language, format=format)
        labels = get_labels(solution[i])
        return features, labels
    g = iter(range(len(data)))
    features_labels_pairs = lambda:map(func, g)
    return features_labels_pairs


def get_output_dim(solution):
    return len(solution[0].split(' '))


def autonlp_2_autodl_format(input_dir: str):
    input_dir = os.path.normpath(input_dir)
    name = os.path.basename(input_dir)
    output_dir = input_dir + '_formatted'

    # Read data
    language = get_language(os.path.join(input_dir, name+'.data', 'meta.json'))
    train_data = read_file(os.path.join(input_dir, name+'.data', 'train.data'))
    train_solution = read_file(os.path.join(input_dir, name+'.data', 'train.solution'))
    test_data = read_file(os.path.join(input_dir, name+'.data', 'test.data'))
    test_solution = read_file(os.path.join(input_dir, name+'.solution'))

    # Create vocabulary
    vocabulary = create_vocabulary(train_data+test_data, language)

    # Convert data into sequences of integers
    features_labels_pairs_train = get_features_labels_pairs(train_data, train_solution,
                                                            vocabulary, language, format=format)
    features_labels_pairs_test = get_features_labels_pairs(test_data, test_solution,
                                                           vocabulary, language, format=format)

    # Write data in TFRecords and vocabulary in metadata
    output_dim = get_output_dim(train_solution)
    col_count, row_count = 1, 1
    sequence_size = -1
    num_channels = 1  # len(vocabulary)
    num_examples_train = len(train_data)
    num_examples_test = len(test_data)
    new_dataset_name = name  # same name
    classes_list = None
    dataset_formatter = UniMediaDatasetFormatter(name,
                                                 output_dir,
                                                 features_labels_pairs_train,
                                                 features_labels_pairs_test,
                                                 output_dim,
                                                 col_count,
                                                 row_count,
                                                 sequence_size=sequence_size, # for strides=2
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
                                                 channels_dict=vocabulary)
    dataset_formatter.press_a_button_and_give_me_an_AutoDL_dataset()
    pass


if __name__ == "__main__":

    if len(argv) == 2:
        input_dir = argv[1]
    else:
        print('Please enter a dataset directory. Usage: `python3 nlp_to_tfrecords path/to/dataset`')
        input_dir = None
        exit()

    autonlp_2_autodl_format(input_dir=input_dir)
