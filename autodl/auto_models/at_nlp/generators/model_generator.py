# -*- coding: utf-8 -*-
# @Date    : 2020/3/3 10:46
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier

from keras_radam import RAdam
from keras.optimizers import SGD, RMSprop, Adamax, Adadelta, Adam

from autodl.utils.log_utils import info
from ...at_nlp.model_manager.emb_utils import generate_emb_matrix
from ...at_nlp.model_lib.cnn_models import TextCNN_Model, CNN_Model
from ...at_nlp.model_lib.rnn_models import TextRCNN_Model, RNN_Model


class ModelGenerator(object):
    def __init__(self,
                 load_pretrain_emb=False,
                 data_feature=None,
                 meta_data_feature=None,
                 fasttext_embeddings_index=None,
                 multi_label=False):

        self.data_feature = data_feature
        self.load_pretrain_emb = load_pretrain_emb
        self.meta_data_feature = meta_data_feature
        self.oov_cnt = 0
        self.embedding_matrix = None
        self.use_bpe = False
        self.multi_label = multi_label
        self.lr = 0.001
        self.cur_lr = 0.0
        self.emb_size = 300
        self.fasttext_embeddings_index = fasttext_embeddings_index

        self.model_lib = {'text_cnn': TextCNN_Model,
                          'text_cnn_2d': CNN_Model,
                          'text_rcnn': TextRCNN_Model,
                          'text_rnn': RNN_Model}

        self.feature_lib = {'char-level + 64dim-embedding', 'char-level + 300dim-embedding',
                            'word-level + pretrained embedding300dim', 'word-level + 64dim-embedding'}

    def select_classifier(self, model_name, feature_mode, data_feature):
        _feature = {}
        if feature_mode == 'char-level + 64dim-embedding':
            _feature = {'use_fasttext_emb': False,
                        'emb_size': 64}
        elif feature_mode == 'char-level + 300dim-embedding':
            _feature = {'use_fasttext_emb': False,
                        'emb_size': 300}
        elif feature_mode == 'word-level + pretrained embedding300dim':
            _feature = {'use_fasttext_emb': True,
                        'emb_size': 300}
        elif feature_mode == 'word-level + 64dim-embedding':
            _feature = {'use_fasttext_emb': False,
                        'emb_size': 64}

        data_feature.update(_feature)
        model = self.build_model(model_name, data_feature=data_feature)
        return model

    def _set_model_compile_params(self, optimizer_name, lr, metrics=[]):
        optimizer = self._set_optimizer(optimizer_name=optimizer_name, lr=lr)
        loss_fn = self._set_loss_fn()
        print(loss_fn)
        if metrics:
            metrics = metrics
        else:
            metrics = ['accuracy']

        return optimizer, loss_fn, metrics

    def _set_model_train_params(self):
        pass

    def build_model(self, model_name, data_feature):
        if model_name == 'svm':
            model = LinearSVC(random_state=0, tol=1e-5, max_iter=500)
            self.model = CalibratedClassifierCV(model)
            if self.multi_label:
                info("use OneVsRestClassifier")
                self.model = OneVsRestClassifier(self.model, n_jobs=-1)

        else:
            if data_feature["use_fasttext_emb"]:
                self.oov_cnt, self.embedding_matrix = self.generate_emb_matrix(
                    num_features=data_feature["num_features"],
                    word_index=data_feature["word_index"])
            else:
                self.embedding_matrix = None

            self.emb_size = data_feature["emb_size"]

            kwargs = {'embedding_matrix': self.embedding_matrix,
                      'input_shape': data_feature['input_shape'],
                      'max_length': data_feature['max_length'],
                      'num_features': data_feature['num_features'],
                      'num_classes': data_feature['num_class'],
                      "filter_num": data_feature["filter_num"],
                      "trainable": False,
                      "emb_size": self.emb_size}
            if self.multi_label:
                kwargs["use_multi_label"] = True

            self.model = self.model_lib[model_name](**kwargs)
            self._set_init_lr(model_name)
            optimizer, loss_fn, metrics = self._set_model_compile_params(optimizer_name='RMSprop',
                                                                         lr=self.lr)
            if self.multi_label:
                loss_fn = 'binary_crossentropy'

            self.model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)

        return self.model

    def _set_loss_fn(self):
        loss_fn = 'categorical_crossentropy'
        return loss_fn

    def _set_optimizer(self, optimizer_name, lr=0.001):
        if optimizer_name == 'RAdam':
            opt = RAdam(learning_rate=lr)
        elif optimizer_name == 'RMSprop':
            opt = RMSprop(lr=lr)
        elif optimizer_name == "Adam":
            opt = Adam(lr=lr)
        return opt

    def _set_init_lr(self, model_name):
        if model_name == "text_cnn":
            self.lr = 0.001
        elif model_name == "text_cnn_2d":
            self.lr = 0.016
        elif model_name == "text_rcnn":
            self.lr = 0.025
        elif model_name == "text_rnn":
            self.lr = 0.0035



    def model_pre_select(self, model_name="svm"):
        self.model_name = model_name

    def generate_emb_matrix(self, num_features, word_index):
        return generate_emb_matrix(num_features=num_features, word_index=word_index,
                                   fasttext_embeddings_index=self.fasttext_embeddings_index)
