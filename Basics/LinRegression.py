import numpy as np
import pandas as pd
import random


metrics = {'mse': lambda pred_y, y: sum((pred_y - y) ** 2) / len(y),
           'mae': lambda pred_y, y: sum(np.abs(pred_y - y)) / len(y),
           'rmse': lambda pred_y, y: np.sqrt(sum((pred_y - y) ** 2) / len(y)),
           'mape': lambda pred_y, y: 100 / len(y) * sum([np.abs((y[i] - pred_y[i]) / y[i]) for i in range(len(y))]),
           'r2': lambda pred_y, y: 1 - sum((y - pred_y) ** 2) / sum([(np.mean(y) - y[i]) ** 2 for i in range(len(y))])
           }

grads = {None: lambda pred_y, y, M, l1_coef, l2_coef, w: (2 / len(y)) * ((pred_y - y) @ M),
         'l1': lambda pred_y, y, M, l1_coef, l2_coef, w: (2 / len(y)) * ((pred_y - y) @ M) + l1_coef * np.sign(w),
         'l2': lambda pred_y, y, M, l1_coef, l2_coef, w: (2 / len(y)) * ((pred_y - y) @ M) + 2 * l2_coef * w,
         'elasticnet': lambda pred_y, y, M, l1_coef, l2_coef, w: (2 / len(y)) * ((pred_y - y) @ M) + l1_coef * np.sign(
             w) + 2 * l2_coef * w
         }


class MyLineReg():
    def __init__(self, n_iter=10, learning_rate=0.5, metric=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=None,
                 random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.last_loss = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self):
        ret = 'MyLineReg class: '
        for i in self.__dict__.keys():
            ret += f"{i}={self.__dict__[i]}, "
        return ret[:-2]

    def fit(self, X, y, verbose=False):
        random.seed(self.random_state)
        M = X
        M.insert(0, 'w0', np.ones(len(M)))
        M = M.to_numpy()
        y = y.to_numpy()
        self.weights = np.ones(X.shape[1])
        for i in range(1, self.n_iter + 1):
            pred_y = M @ self.weights
            loss = (2 / len(X)) * sum((pred_y - y) ** 2)
            if verbose and i % verbose == 0:
                if i != 0:
                    print(f"{i}|loss: {loss}")
                else:
                    print(f"start|loss: {loss}")
            if self.sgd_sample is None:
                grad = grads[self.reg](pred_y, y, M, self.l1_coef, self.l2_coef, self.weights)
            else:
                sgd = self.sgd_sample
                if isinstance(sgd, float):
                    sgd = int(sgd * len(y))
                sample_rows_idx = random.sample(range(len(y)), sgd)
                grad = grads[self.reg](pred_y[sample_rows_idx], y[sample_rows_idx], M[sample_rows_idx], self.l1_coef,
                                       self.l2_coef, self.weights)
            if isinstance(self.learning_rate, (int, float)):
                l_r = self.learning_rate
            else:
                l_r = self.learning_rate(i)
            self.weights = self.weights - l_r * grad
        pred_y = M @ self.weights
        if self.metric is not None:
            self.last_loss = metrics[self.metric](pred_y, y)

    def get_coef(self):
        return self.weights[1:]

    def predict(self, X):
        M = X.copy()
        M.insert(0, 'w0', np.ones(len(X)))
        return M @ self.weights

    def get_best_score(self):
        return self.last_loss
