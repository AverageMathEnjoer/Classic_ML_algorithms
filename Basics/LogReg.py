import numpy as np
import pandas as pd
import random

reg = {None: lambda grad, l1_coef, l2_coef, w: grad,
       'l1': lambda grad, l1_coef, l2_coef, w: grad + l1_coef * np.sign(w),
       'l2': lambda grad, l1_coef, l2_coef, w: grad + 2 * l2_coef * w,
       'elasticnet': lambda grad, l1_coef, l2_coef, w: grad + l1_coef * np.sign(w) + 2 * l2_coef * w
       }


def accuracy(self, y_pred, y_true):
    return (y_true == y_pred).sum() / len(y_true)


def precision(self, y_pred, y_true):
    TP = ((y_pred == 1) & (y_true == 1)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    return TP / (TP + FP)


def recall(self, y_pred, y_true):
    TP = ((y_pred == 1) & (y_true == 1)).sum()
    FN = ((y_pred == 0) & (y_true == 1)).sum()
    return TP / (TP + FN)


def f1(self, y_pred, y_true):
    precision = self.precision(y_pred, y_true)
    recall = self.recall(y_pred, y_true)
    return 2 * precision * recall / (recall + precision)


def roc_auc(self, y_pred, y_true):
    tup = [(y_pred[i], y_true[i]) for i in range(0, len(y_pred))]
    tup.sort(key=lambda tup: tup[0], reverse=True)
    num_of_pos_classes_same = dict()
    p = 0
    n = 0
    for pr, tr in tup:
        if tr == 1:
            if pr in num_of_pos_classes_same:
                num_of_pos_classes_same[pr] += 1
            else:
                num_of_pos_classes_same[pr] = 1
    sum = 0
    for pr, tr in tup:
        if tr == 1:
            p += 1
        else:
            n += 1
            q = 0
            if pr in num_of_pos_classes_same:
                q = num_of_pos_classes_same[pr] / 2.0
            sum += p + q

    return sum * (1.0 / (p * n))


metrics = {'accuracy': accuracy,
           'precision': precision,
           'recall': recall,
           'f1': f1,
           'roc_auc': roc_auc}


class MyLogReg():
    def __init__(self, n_iter=10, learning_rate=0.5, metric=None, reg=None, l1_coef=0, l2_coef=0):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.best_score = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

    def __str__(self):
        ret = 'MyLogReg class: '
        for i in self.__dict__.keys():
            ret += f"{i}={self.__dict__[i]}, "
        return ret[:-2]

    def fit(self, X, y, verbose=False):
        M = X.copy(deep=True)
        M.insert(0, 'w0', np.ones(len(M)))
        M = M.to_numpy()
        y = y.to_numpy()
        self.weights = np.ones(M.shape[1])
        for i in range(self.n_iter):
            pred_y = 1 / (np.exp(-1 * (M @ self.weights)) + 1)
            grad = (1 / len(M)) * ((pred_y - y) @ M)
            grad = reg[self.reg](grad, self.l1_coef, self.l2_coef, self.weights)
            self.weights = self.weights - self.learning_rate * grad
        self.best_score = self.count_metric(M, y)

    def get_coef(self):
        return self.weights[1:]

    def predict_proba(self, X):
        M = X.copy(deep=True)
        M.insert(0, 'w0', np.ones(len(X)))
        pred_y = 1 / (np.exp(-1 * (M @ self.weights)) + 1)
        return pred_y

    def predict(self, X):
        pred_y = self.predict_proba(X)
        return pred_y > 0.5

    def count_metric(self, M, y):
        if self.metric == None:
            return None
        if self.metric == 'roc_auc':
            probs = 1 / (np.exp(-1 * (M @ self.weights)) + 1)
            sc = np.array([probs, y])
            sc = sc[:, sc[0].argsort()]
            eps = 1e-10
            sum = 0.0
            neg, pos = 0, 0
            for i in range(sc.shape[1] - 1, -1, -1):
                if sc[1][i] == 0:
                    for j in range(sc.shape[1] - 1, -1, -1):
                        if sc[1][j] == 1 and sc[0][j] > sc[0][i] + eps:
                            sum += 1
                        elif sc[1][j] == 1 and np.abs(sc[0][j] - sc[0][i]) < eps:
                            sum += 0.5
                    neg += 1
                else:
                    pos += 1
            return sum / (neg * pos)
        else:
            pred_y = (1 / (np.exp(-1 * (M @ self.weights)) + 1)) > 0.5
            tp, tn, fp, fn = 0, 0, 0, 0
            for i in range(len(y)):
                if pred_y[i]:
                    if y[i]:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if y[i]:
                        fn += 1
                    else:
                        tn += 1
            return metrics[self.metric](tp, tn, fp, fn)

    def get_best_score(self):
        return self.best_score
