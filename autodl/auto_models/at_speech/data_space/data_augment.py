import numpy as np

from sklearn.utils import safe_indexing


class DNpAugPreprocessor(object):
    @staticmethod
    def frequency_masking(image, p=0.5, F=0.2):
        _, w, _ = image.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return image

        f = np.random.randint(0, int(w * F))
        f0 = np.random.randint(0, w - f)

        image[:, f0 : f0 + f, :] = 0.0

        return image

    @staticmethod
    def crop_image(image):
        h, w, _ = image.shape
        h0 = np.random.randint(0, h - w)
        image = image[h0 : h0 + w]

        return image


class MixupGenerator(object):
    def __init__(self, X, y, alpha=0.2, batch_size=32, datagen=None, shuffle=True):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.batch_size = batch_size
        self.datagen = datagen
        self.shuffle = shuffle

    def __call__(self):
        while True:
            indices = self.__get_exploration_order()
            n_samples, _, _, _ = self.X.shape
            itr_num = int(n_samples // (2 * self.batch_size))

            for i in range(itr_num):
                indices_head = indices[2 * i * self.batch_size : (2 * i + 1) * self.batch_size]
                indices_tail = indices[(2 * i + 1) * self.batch_size : (2 * i + 2) * self.batch_size]

                yield self.__data_generation(indices_head, indices_tail)

    def __get_exploration_order(self):
        n_samples = len(self.X)
        indices = np.arange(n_samples)

        if self.shuffle:
            np.random.shuffle(indices)

        return indices

    def __data_generation(self, indices_head, indices_tail):
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1_tmp = safe_indexing(self.X, indices_head)
        X2_tmp = safe_indexing(self.X, indices_tail)
        n, _, w, _ = X1_tmp.shape
        X1 = np.zeros((n, w, w, 1))
        X2 = np.zeros((n, w, w, 1))

        for i in range(self.batch_size):
            X1[i] = DNpAugPreprocessor.crop_image(X1_tmp[i])
            X2[i] = DNpAugPreprocessor.crop_image(X2_tmp[i])

        X = X1 * X_l + X2 * (1.0 - X_l)

        y1 = safe_indexing(self.y, indices_head)
        y2 = safe_indexing(self.y, indices_tail)
        y = y1 * y_l + y2 * (1.0 - y_l)

        if self.datagen is not None:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        return X, y


class TTAGenerator(object):
    def __init__(self, X, batch_size):
        self.X = X
        self.batch_size = batch_size

        self.n_samples, _, _, _ = X.shape

    def __call__(self):
        while True:
            for start in range(0, self.n_samples, self.batch_size):
                end = min(start + self.batch_size, self.n_samples)
                X_batch = self.X[start:end]

                yield self.__data_generation(X_batch)

    def __data_generation(self, X_batch):
        n, _, w, _ = X_batch.shape
        X = np.zeros((n, w, w, 1))

        for i in range(n):
            X[i] = DNpAugPreprocessor.crop_image(X_batch[i])

        return X, None
