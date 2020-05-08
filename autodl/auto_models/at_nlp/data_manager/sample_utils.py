import numpy as np
import random


def set_val_ratio(train_label_ratio, train_num):
    if train_label_ratio < 0.01:
        ratio = 0.075
    else:
        ratio = 0.2

    return ratio


def get_sample_index(input_y, num_class):
    # 获取对应label的训练样本的index
    sample_index = []
    for i in range(num_class):
        if list(np.where((input_y[:, i] == 1) == True)[0]) == 0:
            continue
        sample_index.append(
            list(np.where((input_y[:, i] == 1) == True)[0]))
    return sample_index

def downsampling_input_data(input_x, input_y, num_class, max_sample_num):
    sample_index = get_sample_index(input_y, num_class)
    meta_train_index = balance_sampling(sample_index, max_sample_num=max_sample_num, num_classes=num_class)
    random.shuffle(meta_train_index)
    train_sample_x, train_sample_y = map_x_y(meta_train_index, input_x, input_y)
    return train_sample_x, train_sample_y

def balance_sampling(all_index, max_sample_num, num_classes):
    meta_train_index = []
    for i in range(num_classes):  # 按label类别抽取
        if len(all_index[i]) == 0:
            continue
        elif len(all_index[i]) < max_sample_num and len(all_index[i]) > 0:
            tmp = all_index[i] * int(
                max_sample_num / len(all_index[i]))
            tmp += random.sample(all_index[i],
                                 max_sample_num - len(tmp))
            meta_train_index += tmp
        else:
            meta_train_index += random.sample(
                all_index[i], max_sample_num)
    return meta_train_index


def map_x_y(index, x, y):
    _x = [x[i] for i in index]
    _y = y[index, :]
    return _x, _y


def flat_map_x_y(index, x, y):
    flat_index = []
    for i in range(len(index)):
        flat_index.extend(index[i])
    _x, _y = map_x_y(flat_index, x, y)
    return _x, _y


def get_imbalance_statistic(train_label_distribution):
    normal_std = np.std(train_label_distribution) / np.sum(train_label_distribution)
    empty_class_ = [i for i in range(train_label_distribution.shape[0]) if train_label_distribution[i] == 0]
    return normal_std, empty_class_


def do_random_generate_sample(data_x, language, num=1):
    # 默认只随机产生一个样本，减少干扰
    sentence_list = []
    for i in range(num):
        rand_int = random.randint(0, len(data_x) - 1)
        sentence_list.append(data_x[rand_int])
    generate_samples = []
    for sentence in sentence_list:
        if language == "EN":
            words = sentence.split(" ")
        else:
            words = sentence
        if len(words) <= 1:
            new_words = ","
        else:
            w_i = random.randint(0, len(words) - 1)
            new_words = words[w_i]
        generate_samples.append(new_words)
    return generate_samples

def check_x_avg_length(input_x):
    return np.mean([len(x.split(" ")) for x in input_x])