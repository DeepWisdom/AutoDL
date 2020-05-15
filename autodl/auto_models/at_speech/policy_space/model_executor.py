
import numpy as np

from ...at_speech import SLLRLiblinear, SLLRSag, ThinResnet34Classifier
from ...at_speech.data_space.raw_data_space import RawDataNpDb
from ...at_toolkit.at_sampler import AutoSamplerBasic, AutoSpSamplerNew, AutoValidSplitor, minisamples_edaer, sample_y_edaer
from ...at_speech.data_space.feats_data_space import FeatsDataDb
from ...at_speech.policy_space.decision_making import DecisionMaker
from ...at_toolkit.at_tfds_convertor import TfdsConvertor
from ...at_toolkit import AdlClassifier, AdlSpeechDMetadata
from ...at_speech.at_speech_cons import *
from ...at_speech.at_speech_config import TFDS2NP_TAKESIZE_RATION_LIST, TR34_TRAINPIP_WARMUP, IF_VAL_ON, Tr34SamplerHpParams
from autodl.metrics.scores import autodl_auc


CLS_REG_TABLE = {
    CLS_LR_LIBLINEAER: SLLRLiblinear,
    CLS_LR_SAG: SLLRSag,
    CLS_TR34: ThinResnet34Classifier,
}

CLS_2_FEATNAME_REG_TABLE = {
    CLS_LR_LIBLINEAER: FEAT_KAPRE_MELSPECGRAM,
    CLS_LR_SAG: FEAT_KAPRE_MELSPECGRAM,
    CLS_TR34: FEAT_LBS_TR34,
}


class MetaClsHPParams:
    lr_sag_cls_init_params = {"max_iter": 50}


class ModelExecutor:
    def __init__(self, ds_metadata):

        self.class_num = ds_metadata.get("class_num")
        self.train_num = ds_metadata.get("train_num")
        self.test_num = ds_metadata.get("test_num")
        self.aspeech_metadata = AdlSpeechDMetadata(ds_metadata)

        self.cls_tpye_libs = [CLS_LR_LIBLINEAER, CLS_LR_SAG, CLS_TR34]
        self.lr_libl_cls = None
        self.lr_sag_cls = None
        self.tr34_cls = None
        self.tr34_cls_train_pip_run = 0
        self.tfds_convertor = TfdsConvertor()
        self.feats_data_db = FeatsDataDb(self.train_num, self.test_num)

        self.init_pipeline()

        self.train_pip_id = 0
        self.test_pip_id = 0

        self.token_train_size = 0

        self.cur_cls_ins_table = {
            CLS_LR_LIBLINEAER: self.lr_libl_cls,
            CLS_LR_SAG: self.lr_sag_cls,
            CLS_TR34: self.tr34_cls,
        }

        self.decision_maker = DecisionMaker(self.aspeech_metadata)
        self.cur_cls = None
        self.cur_sampler = None
        self.val_sample_idxs = list()
        self.cur_val_examples_y = None
        self.cur_val_nauc = None
        self.cur_train_his_report = dict()

        self.minis_eda_report = None
        self.is_multilabel = False

        self.lr_sampler = AutoSamplerBasic(self.class_num)
        self.tr34_sampler = AutoSpSamplerNew(None)
        self.val_splitor = AutoValidSplitor(self.class_num)

        self.cur_sampler_table = {
            CLS_LR_LIBLINEAER: self.lr_sampler,
            CLS_LR_SAG: self.lr_sampler,
            CLS_TR34: self.tr34_sampler,
        }

        self.tfds2np_take_size_array = TFDS2NP_TAKESIZE_RATION_LIST
        self.tfds2np_takesize_flag = False
        self.decision_maker.infer_model_select_def()
        self.tr34_trainpip_warmup = self.decision_maker.infer_tr34_trainpip_warmup()
        self.tr34_hps_epochs_dict = self.decision_maker.infer_tr34_hps_epoch()
        self.tr34_hps_sample_dict = self.decision_maker.infer_tr34_hps_samplenum()


    def init_pipeline(self):
        self.lr_libl_cls = SLLRLiblinear()
        self.lr_libl_cls.init(self.class_num)

        self.lr_sag_cls = SLLRSag()
        self.lr_sag_cls.init(self.class_num, MetaClsHPParams.lr_sag_cls_init_params)

        self.tr34_cls = ThinResnet34Classifier()
        self.tr34_cls.init(self.class_num)


    def train_pipeline(self, train_tfds, update_train_data=True):
        if self.train_pip_id < len(self.tfds2np_take_size_array):
            if self.train_pip_id == 1:
                take_train_size = max(200, int(self.tfds2np_take_size_array[self.train_pip_id] * self.train_num))
            else:
                take_train_size = int(self.tfds2np_take_size_array[self.train_pip_id] * self.train_num)
        else:
            take_train_size = 200
        self.token_train_size += take_train_size
        self.cur_train_his_report = dict()

        self.tfds_convertor.init_train_tfds(train_tfds, self.train_num)
        if update_train_data is True and self.feats_data_db.raw_data_db.raw_train_np_filled_num < self.train_num:
            accm_raw_train_np_dict = self.tfds_convertor.get_train_np_accm(take_train_size)
            self.minis_eda_report = minisamples_edaer(accm_raw_train_np_dict["x"], accm_raw_train_np_dict["y"])
            if self.minis_eda_report.get("y_cover_rate") <= 0.5:
                self.tfds_convertor.init_train_tfds(train_tfds, self.train_num, force_shuffle=True)
                accm_raw_train_np_dict = self.tfds_convertor.get_train_np_accm(take_train_size)
                self.minis_eda_report = minisamples_edaer(accm_raw_train_np_dict["x"], accm_raw_train_np_dict["y"])

            self.is_multilabel = self.minis_eda_report.get("is_multilabel")
            self.tr34_cls.renew_if_multilabel(self.is_multilabel)

            if self.tfds2np_takesize_flag is False:
                self.decision_maker.learn_train_minisamples_report(self.minis_eda_report)
                self.tfds2np_take_size_array = self.decision_maker.decide_tfds2np_array()
                self.tfds2np_takesize_flag = True

            self.feats_data_db.raw_data_db.put_raw_train_np(accm_raw_train_np_dict["x"], accm_raw_train_np_dict["y"])

        if_split_val = self.decision_maker.decide_if_split_val(self.token_train_size)
        if IF_VAL_ON and if_split_val and len(self.val_sample_idxs) == 0:
            val_mode = "bal"
            val_num = self.decision_maker.decide_g_valid_num()
            self.val_sample_idxs = self.val_splitor.get_valid_sample_idxs(
                np.stack(self.feats_data_db.raw_data_db.raw_train_y_np_table_filled), val_num=val_num, mode=val_mode
            )
            self.feats_data_db.raw_data_db.put_split_valid_np(self.val_sample_idxs)
            self.cur_val_examples_y = self.feats_data_db.get_raw_train_y(self.val_sample_idxs)

        self.cur_cls_name = self.decision_maker.decide_model_select(self.train_pip_id)
        self.cur_cls = self.cur_cls_ins_table.get(self.cur_cls_name)
        self.cur_sampler = self.cur_sampler_table.get(self.cur_cls_name)

        if self.cur_cls_name in [CLS_LR_LIBLINEAER, CLS_LR_SAG]:
            if self.is_multilabel is False:
                self.lr_sampler.init_train_y(self.feats_data_db.raw_data_db.raw_train_y_np_table_filled)
                class_inverted_index_array = self.lr_sampler.init_each_class_index_by_y(self.lr_sampler.train_y)

                cur_train_sample_idxs = self.lr_sampler.init_even_class_index_by_each(class_inverted_index_array)
                cur_train_sample_idxs = [item for sublist in cur_train_sample_idxs for item in sublist]
                cur_train_sample_idxs = [i for i in cur_train_sample_idxs if i not in self.val_sample_idxs]

            else:
                cur_train_sample_idxs = range(len(self.feats_data_db.raw_data_db.raw_train_y_np_table_filled))

            self.cur_feat_name = CLS_2_FEATNAME_REG_TABLE.get(self.cur_cls_name)
            self.use_feat_params = {"len_sample": 5, "sr": 16000}
            cur_train_examples_x = self.feats_data_db.get_raw_train_feats(
                self.cur_feat_name, cur_train_sample_idxs, self.use_feat_params
            )
            cur_train_examples_y = self.feats_data_db.get_raw_train_y(cur_train_sample_idxs)

            train_eda_report = sample_y_edaer(cur_train_examples_y)
            if self.cur_cls_name == CLS_LR_LIBLINEAER:
                assert isinstance(self.cur_cls, SLLRLiblinear), "Error cur_cls is not {}".format(SLLRLiblinear.__name__)
                self.cur_cls.offline_fit(cur_train_examples_x, cur_train_examples_y, fit_params={"if_multilabel": self.is_multilabel})
            elif self.cur_cls_name == CLS_LR_SAG:
                assert isinstance(self.cur_cls, SLLRSag), "Error cur_cls is not {}".format(SLLRSag.__name__)
                self.cur_cls.offline_fit(cur_train_examples_x, cur_train_examples_y, fit_params={"if_multilabel": self.is_multilabel})


        elif self.cur_cls_name in [CLS_TR34]:
            assert isinstance(self.cur_cls, ThinResnet34Classifier), "Error, cls select is {}".format(
                type(self.cur_cls)
            )
            train_use_y_labels = np.stack(self.feats_data_db.raw_data_db.raw_train_y_np_table_filled)

            self.tr34_sampler = AutoSpSamplerNew(y_train_labels=train_use_y_labels)

            if self.is_multilabel is False:
                self.tr34_sampler.set_up()
                cur_train_sample_idxs = self.tr34_sampler.get_downsample_index_list_by_class(
                    per_class_num=Tr34SamplerHpParams.SAMPL_PA_F_PERC_NUM,
                    max_sample_num=self.tr34_hps_sample_dict.get("SAMP_MAX_NUM"),
                    min_sample_num=self.tr34_hps_sample_dict.get("SAMP_MIN_NUM"),
                )
            else:
                cur_train_sample_idxs = self.tr34_sampler.get_downsample_index_list_by_random(
                    max_sample_num=self.tr34_hps_sample_dict.get("SAMP_MAX_NUM"),
                    min_sample_num=self.tr34_hps_sample_dict.get("SAMP_MIN_NUM"))

            cur_train_sample_idxs = [i for i in cur_train_sample_idxs if i not in self.val_sample_idxs]

            self.cur_feat_name = CLS_2_FEATNAME_REG_TABLE.get(self.cur_cls_name)
            if_train_feats_force = self.cur_cls.decide_if_renew_trainfeats()
            self.use_feat_params = self.cur_cls.imp_feat_args
            cur_train_examples_x = self.feats_data_db.get_raw_train_feats(
                self.cur_feat_name, cur_train_sample_idxs, self.use_feat_params, if_train_feats_force
            )
            cur_train_examples_y = self.feats_data_db.get_raw_train_y(cur_train_sample_idxs)
            train_eda_report = sample_y_edaer(cur_train_examples_y)
            self.tr34_cls_train_pip_run += 1
            self.cur_train_his_report = self.cur_cls.online_fit(cur_train_examples_x, cur_train_examples_y, fit_params=self.tr34_hps_epochs_dict)

        if len(self.val_sample_idxs) > 0:
            if self.cur_cls_name == CLS_TR34:
                assert isinstance(self.cur_cls, ThinResnet34Classifier)
                if_force_val_feats = self.cur_cls.decide_if_renew_valfeats()
                use_feat_params = self.cur_cls.imp_feat_args
                cur_val_examples_x = self.feats_data_db.get_split_val_feats(
                    self.cur_feat_name, self.val_sample_idxs, use_feat_params, if_force_val_feats
                )
                cur_val_examples_x = np.array(cur_val_examples_x)
                cur_val_examples_x = cur_val_examples_x[:, :, :, np.newaxis]
                cur_val_preds = self.cur_cls.predict_val_proba(cur_val_examples_x)
            else:
                cur_val_examples_x = self.feats_data_db.get_split_val_feats(self.cur_feat_name, self.val_sample_idxs, self.use_feat_params)
                cur_val_preds = self.cur_cls.predict_proba(cur_val_examples_x, predict_prob_params={"if_multilabel": self.is_multilabel})

            self.cur_val_nauc = autodl_auc(solution=self.cur_val_examples_y, prediction=cur_val_preds)
        else:
            self.cur_val_nauc = -1

        self.train_pip_id += 1
        self.cur_train_his_report["val_nauc"] = self.cur_val_nauc
        self.cur_train_his_report["cls_name"] = self.cur_cls_name
        return self.cur_train_his_report.copy()


    def test_pipeline(self, test_tfds):
        self.tfds_convertor.init_test_tfds(test_tfds)
        if not self.feats_data_db.raw_data_db.if_raw_test_2_np_done:
            raw_test_np = self.tfds_convertor.get_test_np()
            assert isinstance(raw_test_np, list), "raw_test_np is not list"
            self.feats_data_db.raw_data_db.put_raw_test_np(raw_test_np)

        if self.cur_cls_name in [CLS_LR_LIBLINEAER, CLS_LR_SAG]:
            use_feat_params = {"len_sample": 5, "sr": 16000}
            cur_test_examples_x = self.feats_data_db.get_raw_test_feats(self.cur_feat_name, use_feat_params)

            assert isinstance(self.cur_cls, AdlClassifier)
            cur_test_preds = self.cur_cls.predict_proba(cur_test_examples_x, predict_prob_params={"if_multilabel": self.is_multilabel})
            self.test_pip_id += 1
            return np.array(cur_test_preds)

        if self.cur_cls_name in [CLS_TR34]:
            while self.tr34_cls_train_pip_run < self.tr34_trainpip_warmup:
                self.train_pipeline(train_tfds=None, update_train_data=False)

            assert isinstance(self.cur_cls, ThinResnet34Classifier), "Error, cur_cls type error."
            if_force_test_feats = self.cur_cls.decide_if_renew_testfeats()
            use_feat_params = self.cur_cls.imp_feat_args
            cur_test_examples_x = self.feats_data_db.get_raw_test_feats(
                self.cur_feat_name, use_feat_params, if_force_test_feats
            )
            cur_test_examples_x = np.asarray(cur_test_examples_x)

            cur_test_examples_x = cur_test_examples_x[:, :, :, np.newaxis]

            assert isinstance(self.cur_cls, AdlClassifier)

            cur_test_preds = self.cur_cls.predict_proba(cur_test_examples_x)

            del cur_test_examples_x

            self.test_pip_id += 1
            return cur_test_preds


