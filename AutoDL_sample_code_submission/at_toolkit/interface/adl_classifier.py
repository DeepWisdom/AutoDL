import numpy as np


class AdlClassifier(object):
    def init(self, class_num: int, init_params: dict):
        self.class_num = class_num
        self.label_map = list()
        self.clf_name = None
        raise NotImplementedError

    def fit(self, train_examples_x: np.ndarray, train_examples_y: np.ndarray, fit_params:dict):
        raise NotImplementedError

    def predict_proba(self, test_examples: np.ndarray, predict_prob_params: dict) -> np.ndarray:
        raise NotImplementedError

    def rebuild_prob_res(self, input_label_list, orig_prob_array):
        new_prob_arary = np.zeros((orig_prob_array.shape[0], self.class_num))
        for i, cls in enumerate(input_label_list):
            new_prob_arary[:, cls] = orig_prob_array[:, i]

        empty_cls_list = list()
        for i in range(self.class_num):
            if i not in input_label_list:
                empty_cls_list.append(i)

        for sample_i in range(orig_prob_array.shape[0]):
            np_median_value = np.median(new_prob_arary[sample_i])
            for empty_cls in empty_cls_list:
                new_prob_arary[sample_i][empty_cls] = np_median_value

        return new_prob_arary


class AdlOfflineClassifier(AdlClassifier):
    def offline_fit(self, train_examples_x: np.ndarray, train_examples_y: np.ndarray, fit_params:dict):
        raise NotImplementedError


class AdlOnlineClassifier(AdlClassifier):
    def online_fit(self, train_examples_x: np.ndarray, train_examples_y: np.ndarray, fit_params:dict):
        raise NotImplementedError


