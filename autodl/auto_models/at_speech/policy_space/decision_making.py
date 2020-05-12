from ...at_speech.policy_space.meta_learning import ModelSelectLearner
from ...at_speech.policy_space.ensemble_learning import EnsembleLearner
from ...at_toolkit import AdlSpeechDMetadata
from ...at_speech.at_speech_cons import CLS_LR_LIBLINEAER, CLS_LR_SAG, CLS_TR34


class DecisionMaker(object):
    def __init__(self, aspeech_metadata: AdlSpeechDMetadata):
        self.aspeech_metadata = aspeech_metadata
        self.meta_model_select_learner = ModelSelectLearner()
        self.ensemble_learner = EnsembleLearner(self.aspeech_metadata)
        self.aspeech_metadata_minix_report_flag = False

    def learn_train_minisamples_report(self, train_minis_report:dict):
        self.aspeech_metadata.init_train_minisamples_report(train_minis_report)
        self.aspeech_metadata_minix_report_flag = True

    def decide_if_start_val(self):
        self.IF_START_VAL = False

    def decide_if_ensemble_pred(self):
        self.IF_ENSEMBLE_PRED = False

    def decide_model_select(self, train_pip_id):
        return self.meta_model_select_learner.predict_train_cls_select(train_pip_id)

    def decide_g_valid_num(self) -> int:
        return self.ensemble_learner.predict_g_valid_num()

    def decide_if_split_val(self, token_size):
        return self.ensemble_learner.predict_if_split_val(token_size)

    def decide_tfds2np_array(self):
        assert self.aspeech_metadata_minix_report_flag is True, "Error:Meta mini_samples_report flag is False"
        if self.aspeech_metadata.train_minisamples_report.get("x_seqlen_mean") > 200000:
            return [0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        elif self.aspeech_metadata.class_num >= 40:
            return [0.1, 0.2, 0.4, 0.3]
        else:
            return [0.1, 0.2, 0.4, 0.3]


    def infer_model_select_def(self):
        model_select_def = None
        if self.aspeech_metadata.class_num < 10:
            model_select_def = {
                0: CLS_LR_LIBLINEAER,
                1: CLS_LR_LIBLINEAER,
                2: CLS_TR34,
            }
        else:
            model_select_def = {
                0: CLS_LR_LIBLINEAER,
                1: CLS_LR_LIBLINEAER,
                2: CLS_LR_SAG,
                3: CLS_TR34,
            }
        self.meta_model_select_learner.model_select_def = model_select_def

    def infer_tr34_trainpip_warmup(self):
        if self.aspeech_metadata.class_num <= 10:
            return 2
        elif 10 < self.aspeech_metadata.class_num <= 37:
            return 8
        else:
            return 11

    def infer_tr34_hps_epoch(self):
        if self.aspeech_metadata.class_num <= 10:
            first_epoch = 8
        else:
            first_epoch = 14
        left_epoch = 1
        return {"first_epoch": first_epoch, "left_epoch":left_epoch}

    def infer_tr34_hps_samplenum(self):
        tr34_hps_sample_info = dict()
        if self.aspeech_metadata.class_num > 37 or self.aspeech_metadata.train_num > 1000:
            tr34_hps_sample_info["SAMP_MAX_NUM"] = 300
            tr34_hps_sample_info["SAMP_MIN_NUM"] = 300
        else:
            tr34_hps_sample_info["SAMP_MAX_NUM"] = 200
            tr34_hps_sample_info["SAMP_MIN_NUM"] = 200
        return tr34_hps_sample_info

