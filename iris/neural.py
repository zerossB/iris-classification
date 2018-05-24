import numpy as np
import iris.math as mt


class NeuralNetwork:
    def __init__(self, dataset, l_rate):
        self.w = self._init_w(dataset)
        self.l_rate = l_rate
        self.b = np.random.random((dataset.shape[0], 1))
        self.dataset = dataset
        self.iterations = 0
        self.epoch = 0

    def _model(self):
        return self.dataset.dot(self.w) + self.b

    def _init_w(self, dataset):
        row, col = dataset.shape
        return np.random.random((col, col))
