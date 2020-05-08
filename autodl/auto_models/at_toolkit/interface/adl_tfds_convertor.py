#coding:utf-8

class AbsTfdsConvertor:
    def init_train_tfds(self, train_tfds, train_num):
        raise NotImplementedError

    def init_test_tfds(self, test_tfds):
        raise NotImplementedError

    def get_train_np(self, take_size)-> dict:
        raise NotImplementedError

    def get_train_np_accm(self, take_size) -> dict:
        raise NotImplementedError

    def get_train_np_full(self) -> dict:
        raise NotImplementedError

    def get_test_np(self) -> dict:
        raise NotImplementedError



