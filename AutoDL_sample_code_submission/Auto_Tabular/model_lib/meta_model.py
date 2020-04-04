import abc

class MetaModel:
    def __init__(self):
        self.run_num = 0
        self.max_run = None
        self.rise_num = 0
        self.not_rise_num = 0
        self.not_gain_num = 0

        self.all_data_round = None

        self.not_gain_threhlod = 1

        self.auc_gain = 0
        self.best_auc = 0
        self.hist_auc = [0]

        self.best_preds = None

        self.is_init = False

        self.name = None
        self.type = None

    @abc.abstractmethod
    def init_model(self, num_class, **kwargs):
        pass

    @abc.abstractmethod
    def preprocess_data(self, x):
        pass

    @abc.abstractmethod
    def epch_train(self, dataloader, **kwargs):
        pass

    @abc.abstractmethod
    def epoch_valid(self, dataloader):
        pass

    @abc.abstractmethod
    def predict(self, dataloader):
        pass