from __future__ import absolute_import
import os
import numpy as np
import keras
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, LearningRateScheduler

from ...at_toolkit.interface.adl_classifier import AdlOfflineClassifier, AdlOnlineClassifier
from ...at_speech.backbones.thinresnet34 import build_tr34_model
from ...at_speech.at_speech_cons import TR34_PRETRAIN_PATH
from ...at_speech.at_speech_config import ThinRes34Config
from ...at_speech.data_space.examples_gen_maker import DataGenerator as Tr34DataGenerator


IF_TR34_MODELSUMMARY = True


TR34_BB_CONFIG = {
            "gpu": 1,
            "multiprocess": 4,
            "net": "resnet34s",
            "ghost_cluster": 2,
            "vlad_cluster": 8,
            "bottleneck_dim": 512,
            "aggregation_mode": "gvlad",
            "warmup_ratio": 0.1,
            "loss": "softmax",
            "optimizer": "adam",
            "ohem_level": 0,
        }


def set_mp(processes=4):
    import multiprocessing as mp

    def init_worker():
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    global pool
    try:
        pool.terminate()
    except BaseException:
        pass

    if processes:
        pool = mp.Pool(processes=processes, initializer=init_worker)
    else:
        pool = None
    return pool


class TerminateOnBaseline(Callback):
    def __init__(self, monitor="acc", baseline=0.9):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc >= self.baseline:
                print("Epoch %d: Reached baseline, terminating training" % (epoch))
                self.model.stop_training = True


class ThinResnet34Classifier(AdlOnlineClassifier):
    def __init__(self):
        self.tr34_mconfig = ThinRes34Config()
        self.batch_size = self.set_batch_size()

        self.tr34_cls_params = {
            "dim": (257, 250, 1),
            "mp_pooler": set_mp(processes=4),
            "nfft": 512,
            "spec_len": 250,
            "win_length": 400,
            "hop_length": 160,
            "n_classes": None,
            "batch_size": self.batch_size,
            "shuffle": True,
            "normalize": True,
        }

        self.spec_len_status = 0
        self.round_spec_len = self.tr34_mconfig.FE_RS_SPEC_LEN_CONFIG

        self.round_idx = 1
        self.test_idx = 1
        self.g_accept_cur_list = list()
        self.model, self.callbacks = None, None

        self.last_y_pred = None
        self.last_y_pred_round = 0

        self.ml_model = None
        self.is_multilabel = False


    def set_batch_size(self):
        bs = 32
        bs = min(bs, self.tr34_mconfig.MAX_BATCHSIZE)
        return bs

    def step_decay(self, epoch):
        epoch = self.round_idx - 1
        stage1, stage2, stage3 = 10, 20, 40

        if epoch < self.tr34_mconfig.FULL_VAL_R_START:
            lr = self.tr34_mconfig.TR34_INIT_LR
        if epoch == self.tr34_mconfig.FULL_VAL_R_START:
            self.cur_lr = self.tr34_mconfig.STEP_DE_LR
            lr = self.cur_lr
        if epoch > self.tr34_mconfig.FULL_VAL_R_START:
            if self.g_accept_cur_list[-10:].count(False) == 10:
                self.cur_lr = self.tr34_mconfig.MAX_LR
            if self.g_accept_cur_list[-10:].count(True) >= 2:
                self.cur_lr = self.cur_lr * 1.05
            self.cur_lr = max(1e-4 * 3, self.cur_lr)
            self.cur_lr = min(1e-3 * 1.5, self.cur_lr)
            lr = self.cur_lr
        return np.float(lr)

    def tr34_model_init(self, class_num):
        self.tr34_cls_params["n_classes"] = class_num
        model_34 = build_tr34_model(
            net_name='resnet34s',
            input_dim=self.tr34_cls_params["dim"],
            num_class=self.tr34_cls_params["n_classes"],
            tr34_bb_config=TR34_BB_CONFIG
        )

        model = model_34
        if TR34_PRETRAIN_PATH:
            if os.path.isfile(TR34_PRETRAIN_PATH):
                model.load_weights(TR34_PRETRAIN_PATH, by_name=True, skip_mismatch=True)
                if self.tr34_cls_params["n_classes"] >= self.tr34_mconfig.CLASS_NUM_THS:
                    frz_layer_num = self.tr34_mconfig.INIT_BRZ_L_NUM
                else:
                    frz_layer_num = self.tr34_mconfig.INIT_BRZ_L_NUM_WILD
                for layer in model.layers[: frz_layer_num]:
                    layer.trainable = False

            else:
                pass

            pretrain_output = model.output
            weight_decay = self.tr34_mconfig.TR34_INIT_WD

            y = keras.layers.Dense(
                self.tr34_cls_params["n_classes"],
                activation="softmax",
                kernel_initializer="orthogonal",
                use_bias=False,
                trainable=True,
                kernel_regularizer=keras.regularizers.l2(weight_decay),
                bias_regularizer=keras.regularizers.l2(weight_decay),
                name="prediction",
            )(pretrain_output)
            model = keras.models.Model(model.input, y, name="vggvox_resnet2D_{}_{}_new".format("softmax", "gvlad"))
            opt = keras.optimizers.Adam(lr=1e-3)
            model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["acc"])

            ml_y = keras.layers.Dense(
                self.tr34_cls_params["n_classes"],
                activation="sigmoid",
                kernel_initializer="orthogonal",
                use_bias=False,
                trainable=True,
                kernel_regularizer=keras.regularizers.l2(weight_decay),
                bias_regularizer=keras.regularizers.l2(weight_decay),
                name="prediction",
            )(pretrain_output)
            self.ml_model = keras.models.Model(model.input, ml_y, name="vggvox_resnet2D_{}_{}_new".format("sigmoid", "gvlad"))
            ml_opt = keras.optimizers.Adam(lr=1e-3)
            self.ml_model.compile(optimizer=ml_opt, loss="binary_crossentropy", metrics=["acc"])


        if IF_TR34_MODELSUMMARY:
            model.summary()
            self.ml_model.summary()

        callbacks = list()
        if self.tr34_mconfig.ENABLE_CB_ES:
            early_stopping = EarlyStopping(monitor="val_loss", patience=15)
            callbacks.append(early_stopping)
        if self.tr34_mconfig.ENABLE_CB_LRS:
            normal_lr = LearningRateScheduler(self.step_decay)
            callbacks.append(normal_lr)
        return model, callbacks


    def init(self, class_num: int, init_params: dict=None):
        self.class_num = class_num
        self.model, self.callbacks = self.tr34_model_init(class_num)
        self.choose_round_spec_len()

    def try_to_update_spec_len(self):
        if self.round_idx in self.round_spec_len:
            train_spec_len, test_spec_len, suggest_lr = self.round_spec_len[self.round_idx]
            self.update_spec_len(train_spec_len, test_spec_len)

    def choose_round_spec_len(self):
        if self.class_num >= 37:
            self.round_spec_len = self.tr34_mconfig.FE_RS_SPEC_LEN_CONFIG_AGGR
        else:
            self.round_spec_len = self.tr34_mconfig.FE_RS_SPEC_LEN_CONFIG_MILD
        return

    def update_spec_len(self, train_spec_len, test_spec_len):
        self.imp_feat_args = {
            "train_spec_len": train_spec_len,
            "test_spec_len": test_spec_len,
            "train_wav_len": train_spec_len * 160,
            "test_wav_len": test_spec_len * 160,
            "mode": "train",
        }
        self.spec_len_status = 1
        return True

    def decide_if_renew_trainfeats(self):
        # must setup mode.
        self.try_to_update_spec_len()
        self.imp_feat_args["mode"] = "train"
        if self.spec_len_status == 1:
            self.spec_len_status = 2
            return True

    def decide_if_renew_valfeats(self):
        if self.test_idx == 1 or self.spec_len_status == 2:
            self.spec_len_status = 0
            self.imp_feat_args["mode"] = "test"
            return True
        else:
            return False

    def decide_if_renew_testfeats(self):
        if self.test_idx == 1 or self.spec_len_status == 2:
            self.spec_len_status = 0
            self.imp_feat_args["mode"] = "test"
            return True
        else:
            return False

    def decide_epoch_curround(self, first_epoch=14, left_epoch=1):
        if self.round_idx == 1:
            cur_epoch_num = first_epoch
        else:
            cur_epoch_num = left_epoch
        return cur_epoch_num

    def decide_stepperepoch_curround(self, cur_train_len):
        if self.round_idx == 1:
            return max(1, int(cur_train_len // self.tr34_cls_params["batch_size"] // 2))
        else:
            return max(1, int(cur_train_len // self.tr34_cls_params["batch_size"]))

    def renew_if_multilabel(self, is_multilabel=False):
        self.is_multilabel = is_multilabel
        if self.is_multilabel is False:
            pass
        else:
            self.model = self.ml_model

    def online_fit(self, train_examples_x: np.ndarray, train_examples_y: np.ndarray, fit_params:dict):

        self.trn_gen = Tr34DataGenerator(train_examples_x, train_examples_y, **self.tr34_cls_params)
        cur_train_len = len(train_examples_x)
        self.first_r_data_generator = self.trn_gen
        cur_epoch = self.decide_epoch_curround(fit_params.get("first_epoch", 14), fit_params.get("left_epoch", 1))
        early_stopping = TerminateOnBaseline(monitor="acc", baseline=0.999)
        cur_fit_history = self.model.fit_generator(
            self.first_r_data_generator,
            steps_per_epoch=self.decide_stepperepoch_curround(cur_train_len),
            validation_data=fit_params.get("valid_data"), #todo: put in.
            epochs=cur_epoch,
            max_queue_size=10,
            callbacks=self.callbacks + [early_stopping],
            use_multiprocessing=False,
            workers=1,
            verbose=ThinRes34Config.VERBOSE,
        )
        self.round_idx += 1
        cur_train_loss = round(cur_fit_history.history.get("loss")[-1], 6)
        cur_train_acc = round(cur_fit_history.history.get("acc")[-1], 6)
        cur_lr = cur_fit_history.history.get("lr")[-1]
        cur_fit_history_report = {
            "t_loss": cur_train_loss,
            "t_acc": cur_train_acc
        }
        return cur_fit_history_report

    def good_to_predict(self):
        flag = (
            (self.round_idx in self.round_spec_len)
            or (self.round_idx < 10)
            or (self.round_idx < 21 and self.round_idx % 2 == 1)
            or (self.round_idx - self.last_y_pred_round > 3)
        )
        return flag

    def predict_val_proba(self, test_examples: np.ndarray, predict_prob_params: dict = None) -> np.ndarray:
        K.set_learning_phase(0)
        y_pred = self.model.predict(test_examples, batch_size=self.tr34_mconfig.PRED_SIZE)
        return y_pred

    def predict_proba(self, test_examples: np.ndarray, predict_prob_params: dict=None) -> np.ndarray:
        K.set_learning_phase(0)
        if self.good_to_predict():
            y_pred = self.model.predict(test_examples, batch_size=self.tr34_mconfig.PRED_SIZE)
            self.test_idx += 1

            self.last_y_pred = y_pred
            self.last_y_pred_round = self.round_idx
            return y_pred
        else:
            return self.last_y_pred

    def predict_proba_2(self, test_examples: np.ndarray, predict_prob_params: dict=None) -> np.ndarray:
        K.set_learning_phase(0)
        y_pred = self.model.predict(test_examples, batch_size=self.tr34_mconfig.PRED_SIZE)
        self.test_idx += 1
        return y_pred

