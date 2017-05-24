# -*- coding:utf-8 -*-


def class_reblance(c, type_=1):
    '''
    solve the problem of the class imbalance
    -----
    params:
        X: the datasets
        c: the classes
        type_: 
            1- 估计正例比率/(1-估计正例比率) * 数据集分布反例数/数据集分布正例数
            2- 欠采样：即采集和反例数相当的正例数
            3- 集成方法：将正例数分成多个和反例数相当的份数，然后每一份和反例进行算法拟合，然后最后使用集成方法来得到最后结果
            4- 代价敏感：将上面的数据集分布的正反例数比值换做代价比例，即将正类分为反类的代价/反类分为正类的代价
    '''
    unique_class = list(set(c.flatten().tolist()))

    if len(unique_class) > 2:
        raise Exception('the number of the classes must be 2')

    c_1 = c[c == unique_class[0]]
    c_2 = c[c == unique_class[1]]
    c_1 == c_2
