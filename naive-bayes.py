import numpy as np
import itertools as it

class NaiveBayes:
    def __init__(self, data, classes=2):
        self._class_cnt = classes
        self._attr_cnt = len(data[0,:]) - 1
        self._classes = list(range(self._class_cnt))
        self._attrs = list(range(self._attr_cnt))
        self._vars = np.zeros((classes, self._attr_cnt), dtype='float32')
        self._avgs = np.zeros((classes, self._attr_cnt), dtype='float32')
        self._prior = np.zeros((self._class_cnt), dtype='float32')

        self.train(data)


    def train(self, data):
        for k in self._classes:
            data_k = np.array([x for x in data if x[0] == k])

            for i in self._attrs:
                self._vars[k, i] = np.var(data_k[:,i+1])
                self._avgs[k, i] = np.mean(data_k[:,i+1])

            self._prior[k] = len([x for x in data if x[0]==k]) / len(data)


    def predict(self, x):
        def prod(x, k):
            acc = 1
            for i in self._attrs:
                acc *= self.posterior(list(x)[i], i, k)
            return acc

        return np.argmax([prod(x, k)*self._prior[k] for k in self._classes])


    def posterior(self, a_i, i, k):
        var = self._vars[k, i]
        avg = self._avgs[k, i]

        return ((2*np.pi*var)**-0.5 * (np.exp((-(a_i - avg)**2) / (2*var))))
