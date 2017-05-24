# -*- coding:utf-8 -*-
import matplotlib


class ROC(object):
    def __init__(self):
        pass

    '''
    calculate the roc vlaue
    -------
    params:
        c: the classes of the datasets
        yc: the prediction results of the datasets
    '''
    def calcu_roc(self, X, c, module):

