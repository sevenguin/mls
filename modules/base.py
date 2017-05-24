# -*- coding:utf-8 -*-


class BaseModule(object):
    def __init__(self):
        pass
    
    def fit(self, *args, **argw):
        raise Exception('BaseModule class is an abstract class')

    def predict(self, *args, **argw):
        raise Exception('BaseModule class is an abstract class')

