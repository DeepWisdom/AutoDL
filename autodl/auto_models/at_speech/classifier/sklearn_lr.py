from __future__ import absolute_import
import numpy as np

from sklearn.linear_model import logistic, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from ...at_toolkit.interface.adl_classifier import AdlOfflineClassifier, AdlOnlineClassifier


def ohe2cat(label):
    return np.argmax(label, axis=1)


class SLLRLiblinear(AdlOfflineClassifier):

    def init(self, class_num: int, init_params: dict = None):
        self.clf_name = "sl_lr_liblinear"
        self.class_num = class_num
        self.model = logistic.LogisticRegression(solver="liblinear")
        self.ml_mode = 2
        self.ml_models = [OneVsRestClassifier(logistic.LogisticRegression(solver="liblinear")) for i in range(class_num)]
        self.ml_model = OneVsRestClassifier(logistic.LogisticRegression(solver="liblinear"))
        self.logReg_pipeline = Pipeline([('clf', OneVsRestClassifier(logistic.LogisticRegression(solver='liblinear'), n_jobs=-1)),])

    def offline_fit(self, train_examples_x: np.ndarray, train_examples_y: np.ndarray, fit_params: dict = None):
        if fit_params.get("if_multilabel") is False:
            train_examples_y = ohe2cat(train_examples_y)
            self.model.fit(train_examples_x, train_examples_y)
            self.label_map = self.model.classes_
        else:
            if self.ml_mode == 1:
                for cls in range(self.class_num):
                    cls_y = train_examples_y[:, cls]
                    self.ml_models[cls].fit(train_examples_x, cls_y)

            elif self.ml_mode == 2:
               self.ml_model.fit(train_examples_x, train_examples_y)

            elif self.ml_mode == 3:
                for cls in range(self.class_num):
                    cls_y = train_examples_y[:, cls]
                    self.logReg_pipeline.fit(train_examples_x, cls_y)

            else:
                pass

    def predict_proba(self, test_examples: np.ndarray, predict_prob_params: dict = None) -> np.ndarray:
        if predict_prob_params.get("if_multilabel") is True:
            return self.predict_proba_multilabel(test_examples)

        else:
            raw_pred_probas = self.model.predict_proba(test_examples)
            if len(self.label_map) < self.class_num:
                rebuilt_pred_proba = self.rebuild_prob_res(self.label_map, raw_pred_probas)
                return rebuilt_pred_proba
            else:
                return raw_pred_probas

    def predict_proba_multilabel(self, test_examples: np.ndarray):
        if self.ml_mode == 1:
            all_preds = []
            for cls in range(self.class_num):
                preds = self.ml_models[cls].predict_proba(test_examples)
                all_preds.append(preds[:, 1])

            preds = np.stack(all_preds, axis=1)

        elif self.ml_mode == 2:
            preds = self.ml_model.predict_proba(test_examples)

        elif self.ml_mode == 3:
            all_preds = []
            for cls in range(self.class_num):
                preds = self.logReg_pipeline.predict_proba(test_examples)
                all_preds.append(preds[:, 1])

            preds = np.stack(all_preds, axis=1)

        else:
            preds = self.ml_model.predict_proba(test_examples)

        return preds


class SLLRSag(AdlOfflineClassifier):

    def init(self, class_num, init_params: dict):
        self.clf_name = "sl_lr_sag"
        self.class_num = class_num
        self.max_iter = init_params.get("max_iter")
        self.model = logistic.LogisticRegression(C=1.0, max_iter=self.max_iter, solver="sag", multi_class="auto")

        self.ml_model = OneVsRestClassifier(logistic.LogisticRegression(solver="liblinear"))

    def offline_fit(self, train_examples_x: np.ndarray, train_examples_y: np.ndarray, fit_params: dict = None):
        if fit_params.get("if_multilabel") is False:
            train_examples_y = ohe2cat(train_examples_y)
            self.model.fit(train_examples_x, train_examples_y)
            self.label_map = self.model.classes_

        else:
            self.ml_model.fit(train_examples_x, train_examples_y)

    def predict_proba(self, test_examples: np.ndarray, predict_prob_params: dict = None) -> np.ndarray:
        if predict_prob_params.get("if_multilabel") is True:
            return self.predict_proba_multilabel(test_examples)

        else:
            raw_pred_probas = self.model.predict_proba(test_examples)
            if len(self.label_map) < self.class_num:
                rebuilt_pred_proba = self.rebuild_prob_res(self.label_map, raw_pred_probas)
                return rebuilt_pred_proba
            else:
                return raw_pred_probas


    def predict_proba_multilabel(self, test_examples: np.ndarray, predict_prob_params: dict = None) -> np.ndarray:
        preds = self.ml_model.predict_proba(test_examples)
        return preds


class MLLRLiblinear(AdlOfflineClassifier):
    def init(self, class_num: int, init_params: dict):
        self.clf_name = "ml_sl_lr_liblinear"
        self.class_num = class_num
        self.model = logistic.LogisticRegression(solver="liblinear")

    def offline_fit(self, train_examples_x: np.ndarray, train_examples_y: np.ndarray, fit_params:dict):
        pass

    def predict_proba(self, test_examples: np.ndarray, predict_prob_params: dict) -> np.ndarray:
        pass


def main():
    class_num = 100
    lr_libl_cls_init_params = {}
    lr_sag_cls_init_params = {"max_iter": 30}  # 50/100
    lr_libl_cls = SLLRLiblinear()
    lr_libl_cls.init(class_num)

    lr_sag_cls = SLLRSag()
    lr_sag_cls.init(class_num, lr_sag_cls_init_params)


if __name__ == "__main__":
    main()
