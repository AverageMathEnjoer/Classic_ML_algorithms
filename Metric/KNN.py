import numpy as np
import pandas as pd
import random

distance = {'euclidean': lambda x, y: np.sqrt(sum((x - y) ** 2)),
            'chebyshev': lambda x, y: max(np.abs(x - y)),
            'manhattan': lambda x, y: sum(np.abs(x - y)),
            'cosine': lambda x, y: 1 - np.dot(x, y) / np.sqrt(sum(x ** 2) * sum(y ** 2))
            }

dot_weigth = dict(uniform=lambda y, near, dist: sum(y[near]) / len(near),
                  rank=lambda y, near, dist: sum(y[near] * np.array([1 / (i) for i in range(1, len(near) + 1)])) / sum(
                      np.array([1 / (i) for i in range(1, len(near) + 1)])),
                  distance=lambda y, near, dist: sum(y[near] * 1 / dist[near]) / sum(1 / dist[near]))


class MyKNNClf:
    def __init__(self, k=3, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.train_size = None

    def __str__(self):
        ret = 'MyKNNClf class: '
        for i in self.__dict__.keys():
            ret += f"{i}={self.__dict__[i]}, "
        return ret[:-2]

    def fit(self, X, y):
        self.X = X.to_numpy()
        self.y = y.to_numpy()
        self.train_size = X.shape

    def predict_proba(self, T):
        T = T.copy(deep=True)
        T = T.to_numpy()
        probs = np.zeros(len(T))
        for i in range(len(T)):
            dist = np.zeros(len(self.X))
            for j in range(len(self.X)):
                dist[j] = distance[self.metric](T[i], self.X[j])
            near = dist.argsort()[:self.k]
            probs[i] = dot_weigth[self.metric](self.y, near, dist)
        return probs

    def predict(self, T):
        probs = self.predict_proba(T)
        return probs >= 0.5


class MyKNNReg:
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.metric = metric
        self.train_size = None
        self.weight = weight

    def __str__(self):
        ret = 'MyKNNReg class: '
        for i in self.__dict__.keys():
            ret += f"{i}={self.__dict__[i]}, "
        return ret[:-2]

    def fit(self, X, y):
        self.X = X.to_numpy()
        self.y = y.to_numpy()
        self.train_size = X.shape

    def predict(self, T):
        T = T.copy(deep=True)
        T = T.to_numpy()
        pred_y = np.zeros(len(T))
        for i in range(len(T)):
            dist = np.zeros(len(self.X))
            for j in range(len(self.X)):
                dist[j] = distance[self.metric](T[i], self.X[j])
            near = dist.argsort()[:self.k]
            pred_y[i] = dot_weigth[self.weight](self.y, near, dist)
        return pred_y
