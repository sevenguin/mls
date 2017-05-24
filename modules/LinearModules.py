# -*- coding:utf-8 -*-
'''
the algorithms of Linear Regression
auth: sevenguin <scuwei@hotmail.com>
date: 20170523
'''
import numpy as np
import base


class BaseLR(base.BaseModule):
    '''
    base linearregression
    ------------
    params: w,b the init value of the regression's params
    '''

    def _trans_to_nparray(self, datasets):
        if type(datasets) is not np.array:
            datasets = np.array(datasets)
        return datasets


class LinearRegression(BaseLR):
    '''
    params:
       k: 迭代次数
       gd_type: batch/stochastic
    '''
    def __init__(self):
        pass

    def _batch_gradient_descent(self):
        for i in range(self.k):
            deltw = np.zeros(self.n + 1)
            for j, x in enumerate(self.X):
                deltw += (np.dot(self.w, x) - self.y[j]) * x * self.alhpa
            self.w = self.w - deltw

    def fit(self, X, y, k=20, w=None, alpha=0.1, gd_type='batch'):
        self.X = self._trans_to_nparray(X)
        self.y = self._trans_to_nparray(y)

        self.m, self.n = self.X.shape
        if self.m != self.y.shape[0]:
            raise Exception('wrong shape')

        self.k, self.w = k, w
        if not self.w:
            self.w = np.ones(self.n + 1)
        self.X = np.c_[self.X, np.ones(self.m)]

        self.alhpa = 0.1
        try:
            self.w = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(y)
        except Exception:
            self._batch_gradient_descent()
        self.w = [round(n, 5) for n in self.w]

    def predict(self, x):
        return np.dot(np.c_(x, [1]), self.w)


class LogisticRegression(BaseLR):
    def __init__(self):
        pass
