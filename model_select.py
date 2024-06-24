# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:33:58 2022

@author: Yang, Zeyuan
该脚本用于初步筛选模型
"""

import numpy as np
import virtual_samples as vs
import random
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor #实现部分模型的多输出回归

from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error, mean_absolute_percentage_error, r2_score
from Evalute_metrics import RMSE, AP, MAP
# 评估指标介绍https://www.cnblogs.com/mdevelopment/p/9456486.html
# EV: 解释回归模型的方差得分，[0,1]，接近1说明自变量越能解释因变量的方差变化
# MAE: 平均绝对误差，评估预测结果和真实数据集的接近程度的程度，越小越好
# MSE: 均方差， 计算拟合数据和原始数据对应样本点的误差的平方和的均值，越小越好
# R2: 判定系数，解释回归模型的方差得分，[0,1]，接近1说明自变量越能解释因变量的方差变化。

# =============================================================================
# 数据预处理
X_train, X_test, y_train, y_test = vs.VSG_ID( 
                                        virtual= True, 
                                        affi=0.99, 
                                        vir_num = 0, 
                                        method = 2, 
                                        seed1=42
                                             )
# =============================================================================

model_br = BayesianRidge()  # 建立贝叶斯回归模型
model_rm = RandomForestRegressor() # 建立随机森林回归
model_etc = ElasticNet() # 建立弹性网络回归模型
model_svr = SVR() # 建立支持向量回归模型
model_gbr = GradientBoostingRegressor(loss ='huber',
                                      learning_rate = 0.01,
                                      n_estimators = 50,
                                      max_depth= 10) # 建立梯度增强回归模型
determined_params= {
                'n_estimators': 120,
                'max_depth': 3,
                'subsample': 0.9,
                'min_child_weight': 0.01,
                'learning_rate': 0.1,
                'n_jobs': -1,
                'alpha':0,
                'lambda':1,
                'gamma':0.01,
                }

model_xgb = xgb.XGBRegressor(**determined_params) # XGBoost回归模型
model_kn = KNeighborsRegressor()

model_names = ['BayesianRidge', 'SVR', 'KNeighbors',
               'RandomForest','GBR', 'XGBR']
model_dir = [model_br,model_svr,model_kn, model_rm, model_gbr, model_xgb]

wrapper_dir = []
for model in model_dir:
    wrapper_dir.append(MultiOutputRegressor(model))
# =============================================================================
# 交叉验证评分与模型训练
n_folds = 6 # 设置交叉检验的次数
cv_score_list = [] # 交叉检验结果列表
y_train_pre = [] # 各个模型预测的y值列表
y_test_pre = [] # 创建测试集预测结果列表

for wrapper in wrapper_dir:
# =============================================================================
#     # cross_val_score()主要是用来测试这个模型在这个数据集合上的表现，
#     # 并不是一个生成训练模型的方法，更准确地说是一个评价模型的方法。
#     # 若希望实现基于交叉验证的训练，可以采用ensemble model 将交叉验证中的模型集成。
#     scores = cross_val_score(wrapper, X_train, y_train, 
#                              cv = n_folds, scoring = 'r2')
#     # 对每个回归模型进行交叉验证
#     cv_score_list.append(scores) # 将验证结果保存在列表中
# =============================================================================

    # 模型训练
    y_train_pre.append(wrapper.fit(X_train, y_train).predict(X_train))
    y_test_pre.append(wrapper.fit(X_train, y_train).predict(X_test))
    # 将训练模型的预测结果保存在列表中

    
# ============================================================================= 
# 模型拟合与预测效果评估
n_samples, n_features = X_train.shape # 总训练样本量，总特征量
n_samples_test = X_test.shape[0]

wrapper_metrics_name = [mean_absolute_error, 
                      RMSE, mean_absolute_percentage_error, r2_score]
wrapper_train_metrics = [] # 回归训练评价指标列表
wrapper_test_metrics = [] # 回归预测评价指标列表

for i in range(len(wrapper_dir)):
    tmp_list = [] 
    tmp2_list = []
    for m in wrapper_metrics_name:
        tmp_score = m(y_train, y_train_pre[i]) # 计算每个回归指标结果
        tmp_list.append(tmp_score)
        tmp2_score = m(y_test, y_test_pre[i])
        tmp2_list.append(tmp2_score)
        
    wrapper_train_metrics.append(tmp_list)
    wrapper_test_metrics.append(tmp2_list)
    
df1 = pd.DataFrame(cv_score_list, index = model_names) # 建立交叉检验评分数据框
df2 = pd.DataFrame(wrapper_train_metrics, index = model_names, 
                   columns = ['MAE', 'RMSE', 'MAPE','R2']) # 建立回归训练评估数据框
df3 = pd.DataFrame(wrapper_test_metrics, index = model_names, 
                   columns = ['MAE', 'RMSE','MAPE', 'R2']) # 建立回归预测评估数据框

print('samples: %d \t features: %d' %(n_samples, n_features))

print(70*'-') # 打印分割线
print('Cross validation result:')
print(df1)

print(70*'-')
print('Regression train metrics:')
print(df2)

print(70*'-')
print('Regression test metrics:')
print(df3)

print(70*'-')
# =============================================================================
# print('short name \t full name')
# print('EV \t explained_variance')
# print('MAE \t mean_absolute_error')
# print('MSE \t mean_squared_error')
# print('R2 \t R2')
# print(70*'-')
# =============================================================================

# =============================================================================
# 模型拟合效果可视化
plt.figure(num = 'Train', figsize = (16,12), edgecolor = 'k', frameon = True)
plot_list = {'markersize': 8, 'linewidth': 2} # 统一设置标记点大小和线宽
linestyle_list = ['r-+', 'g-o', 'b-*', 'y-^', 'c-v', 'm-x','w-p'] # 线条颜色及样式列表
for j in range(y_train.shape[1]):
    
    plt.subplot(4,2,j+1)
    plt.plot(np.arange(n_samples), y_train[:,j], color = 'k', label='True y',**plot_list) # 真实标签
    for i, pre_y in enumerate(y_train_pre):
        plt.plot(np.arange(n_samples), pre_y[:,j], 
                 linestyle_list[i], label = model_names[i],**plot_list)
    
    font1 = {'family':'Time New Roman', 'weight':'normal', 'size':12}
    font2 = {'family':'Time New Roman', 'weight':'normal', 'size':8}
    
    plt.title('Regression result comparsion', font1)
    plt.legend(loc = 'upper right', prop = font2)
    plt.xlabel('Number', font1)
    plt.ylabel('Real and prediction values (μm)', font1)

    #设置坐标刻度值的大小
    plt.tick_params(labelsize=23)
    plt.grid()  # 生成网格
    
# =============================================================================
# 模型评估可视化
plt.figure(num = 'Train_evaluate', figsize = (16,12), edgecolor = 'k', frameon = True)

absolute_deviation_train = np.empty([y_train.shape[0],len(wrapper_dir)])
for i in range(len(y_train_pre)): 
    absolute_deviation_train[:,i] = np.sum((y_train_pre[i]-y_train)**2,axis = 1)
    plt.plot(np.arange(n_samples),absolute_deviation_train[:,i],
             linestyle_list[i], label = model_names[i], **plot_list)
    
font1 = {'family':'Time New Roman', 'weight':'normal', 'size':24}
font2 = {'family':'Time New Roman', 'weight':'normal', 'size':12}

plt.legend(loc = 'upper right', prop = font2)
plt.xlabel('Number', font1)
plt.ylabel('Absolute deviation (μm)', font1)
plt.tick_params(labelsize=23)
plt.grid()  # 生成网格
# =============================================================================
# 模型预测评估与可视化
plt.figure(num = 'Prediction', figsize = (16,12), edgecolor = 'k', frameon = True)
for j in range(y_test.shape[1]):
    plt.subplot(4,2,j+1) 
    plt.plot(np.arange(n_samples_test), y_test[:,j], color = 'k', label='True y', **plot_list) # 真实标签
    for i, pre_y in enumerate(y_test_pre):
        plt.plot(np.arange(n_samples_test), pre_y[:,j], 
                 linestyle_list[i], label = model_names[i],**plot_list)

    font1 = {'family':'Time New Roman', 'weight':'normal', 'size':24}
    font2 = {'family':'Time New Roman', 'weight':'normal', 'size':12}
    
    plt.title('Regression result comparsion', font1)
    plt.legend(loc = 'upper right', prop = font2)
    plt.xlabel('Number', font1)
    plt.ylabel('Real and prediction values (μm)', font1)

    #设置坐标刻度值的大小
    plt.tick_params(labelsize=23)  
    plt.grid()  # 生成网格  
# =============================================================================
# 模型预测评估可视化
plt.figure(num = 'Prediction_evaluate', figsize = (16,12), edgecolor = 'k', frameon = True)

absolute_deviation_test = np.empty([y_test.shape[0],len(wrapper_dir)])
for i in range(len(y_test_pre)): 
    absolute_deviation_test[:,i] = np.sum((y_test_pre[i]-y_test)**2,axis = 1)
    plt.plot(np.arange(n_samples_test),absolute_deviation_test[:,i],
             linestyle_list[i], label = model_names[i], **plot_list)
    
font1 = {'family':'Time New Roman', 'weight':'normal', 'size':12}
font2 = {'family':'Time New Roman', 'weight':'normal', 'size':8}

plt.legend(loc = 'upper right', prop = font2)
plt.xlabel('Number', font1)
plt.ylabel('Absolute deviation (μm)', font1)
plt.tick_params(labelsize=23)
plt.grid()  # 生成网格
