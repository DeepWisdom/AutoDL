# coding:utf-8


class AdlSpeechDMetadata:
    def __init__(self, raw_metadata:dict):
        self.train_num = raw_metadata.get("train_num")
        self.test_num = raw_metadata.get("test_num")
        self.class_num = raw_metadata.get("class_num")
        self.train_minisamples_report = None

    def init_train_minisamples_report(self, minispls_report:dict):
        self.train_minisamples_report = minispls_report
