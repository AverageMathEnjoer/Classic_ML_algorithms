import numpy as np
import pandas as pd
import random

eps = 1e-10


distance = {'euclidean': lambda x, y: np.sqrt(sum((x - y) ** 2)),
            'chebyshev': lambda x, y: max(np.abs(x - y)),
            'manhattan': lambda x, y: sum(np.abs(x - y)),
            'cosine': lambda x, y: 1 - np.dot(x, y) / np.sqrt(sum(x ** 2) * sum(y ** 2))
            }


def dist(x, ct):
    return sum((x - ct) ** 2)


class MyKMeans:
    def __init__(self, n_clusters=3, max_iter=10, n_init=3, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    def __str__(self):
        ret = 'MyKMeans class: '
        for i in self.__dict__.keys():
            ret += f"{i}={self.__dict__[i]}, "
        return ret[:-2]

    def fit(self, X):
        np.random.seed(self.random_state)

        def random_centers():
            centroids = []
            for i in range(self.n_clusters):
                cnt = np.zeros(len(X.columns))
                pos = 0
                for j in X.columns:
                    j_coord = np.random.uniform(X[j].min(), X[j].max())
                    cnt[pos] += j_coord
                    pos += 1
                centroids.append(cnt)
            return centroids

        def count_clusters(centroids):
            wcss = 0
            y = np.zeros(len(X))
            for i in range(len(X)):
                min_d = np.Inf
                for ct_num in range(len(centroids)):
                    if dist(X.iloc[i], centroids[ct_num]) < min_d:
                        min_d = dist(X.iloc[i], centroids[ct_num])
                        y[i] = ct_num
                wcss += min_d
            return y, wcss

        def count_centers(y):
            new_centers = [0] * self.n_clusters
            for i in range(self.n_clusters):
                new_centers[i] = np.array(X[y == i].mean())

            return new_centers

        def Kmean_algo():
            centroids = random_centers()
            for iter in range(self.max_iter):
                clusters, wcss = count_clusters(centroids)
                new_centers = count_centers(clusters)
                if all([dist(new_centers[i], centroids[i]) < eps for i in range(len(new_centers))]):
                    break
                for i in range(len(centroids)):
                    if not any(np.isnan(new_centers[i])):
                        centroids[i] = new_centers[i]
            return centroids, wcss

        self.inertia_ = np.Inf
        for init in range(self.n_init):
            cnt, wcss = Kmean_algo()
            if wcss < self.inertia_:
                self.inertia_ = wcss
                self.cluster_centers_ = cnt

    def predict(self, X):
        pred_y = np.zeros(len(X))
        for i in range(len(X)):
            min_d = np.Inf
            for j in range(len(self.cluster_centers_)):
                if dist(X.iloc[i], self.cluster_centers_[j]) < min_d:
                    min_d = dist(X.iloc[i], self.cluster_centers_[j])
                    pred_y[i] = j
        return pred_y
