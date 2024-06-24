# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 16:02:48 2022

@author: Yang, Zeyuan
采用GridSearchCV对XGBRegreesor()调参

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import virtual_samples as vs
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor #实现部分模型的多输出回归
from sklearn.metrics import explained_variance_score,mean_absolute_error,\
    mean_squared_error, mean_absolute_percentage_error, r2_score
    
from Evalute_metrics import RMSE, AP, MAP
import xgboost as xgb
import winsound
# =============================================================================
# 样本处理
wrapper_metrics_name = [mean_absolute_error, mean_squared_error, r2_score]
metrics_name = ['MAE','MSE','R2']

X_train, X_test, y_train, y_test = vs.VSG_ID( 
                                        virtual= True, 
                                        affi=0.99, 
                                        vir_num = 55, 
                                        method = 2, 
                                        seed1=42
                                             )

# =============================================================================
# XGBR调参
# 因为后面采用了MultiOutputRegressor()，模型的参数前应该添加estimator__，才能正确输入参数
params_gbr = {  #https://xgboost.readthedocs.io/en/latest/parameter.html#general-parameters
                #'estimator__n_estimators':[200,300,400,500,1000,2000],
                #树的最大深度，值越大，越能学到更具体局部的样本
                #'estimator__max_depth':[2,3,4,5,7,8,9,10,15],
                #叶子节点样本权重和，用于避免过拟合，值过大会导致欠拟合
                #'estimator__subsample':[0.7,0.8,0.85,0.9,0.95,0.97,1],#防止过拟合(0,1]，最优为1
                #(0,1],subsample ration of columns when constructing each tree.
                #'estimator__colsample_bytree':[0.3,0.4,0.45,0.5,0.55,0.6,0.7],
                #'estimator__gamma':[0,0.0001,0.001,0.002,0.004],#越大算法越保守，最优为0，05
                #'estimator__alpha':[0,0.04,0.03,0.05,0.1,0.5],#L1 正则项，越大越保守，默认为0
                #'estimator__lambda':[0,0.01,1,2,3],#L2 正则项，越大越保守，默认为1，最优为1
                'estimator__learning_rate':[0.01,0.06,0.07,0.08,0.1,0.2,0.3,0.4,0.5]
                }

determined_params= {
                'learning_rate': 0.01,
                'n_estimators': 2000,
                'subsample': 0.7,
                'max_depth': 5,
                'min_samples_split':2,
                'min_weight_fraction_leaf':0,
                'min_samples_leaf':1,
                'alpha':0.9,
                'random_state':4,
                }

model_gbr = GradientBoostingRegressor(**determined_params) # 建立梯度增强回归模型 model_xgb = xgb.XGBRegressor() # 建立梯度增强回归模型 
wrapper_xgb =  MultiOutputRegressor(model_gbr,n_jobs = -1)


crossed_gbr = GridSearchCV(wrapper_xgb, params_gbr, cv = 5,
                                 scoring = 'neg_mean_squared_error', n_jobs = -1)


crossed_gbr.fit(X_train, y_train)
result = crossed_gbr.cv_results_
print(70*'=')
print('GBDT best score is :' , crossed_gbr.best_score_)
print('GBDT best parameters are:', crossed_gbr.best_params_)
print('\n')
# =============================================================================
y_test_pre = crossed_gbr.predict(X_test)
scores = [mean_absolute_error, mean_squared_error, r2_score]
scores_value = []
for m in scores:
    scores_value.append(m(y_test, y_test_pre)) # 计算每个回归指标结果
    
winsound.Beep(3000, 440) #提示音