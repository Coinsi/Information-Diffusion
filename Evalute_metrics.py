# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:05:09 2022

@author: Yang, Zeyuan
评估指标
"""

import numpy as np
from sklearn.metrics import mean_squared_error 

def RMSE(y_test,y_pre):
    score_rmse = np.sqrt(mean_squared_error(y_test, y_pre))
    return score_rmse

def AP(y_test,y_pre):
    score_AP = np.empty([y_test.shape[0],1])
    score_AP = np.sum((y_test-y_pre)**2,1)
    return score_AP

def MAP(y_test, y_pre):
    score_MAP = np.mean(np.sum((y_test-y_pre)**2,1))
    return score_MAP