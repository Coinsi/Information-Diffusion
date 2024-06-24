# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 20:12:08 2022

@author: Surface
用于比较有无虚拟样本对不同模型的提升效果
"""
import virtual_samples as vs
import pandas as pd
import numpy as np
import dataprocess as dp
from sklearn.model_selection import train_test_split


from sklearn.linear_model import BayesianRidge, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor #实现部分模型的多输出回归

from sklearn.metrics import mean_absolute_error,\
    mean_squared_error, mean_absolute_percentage_error, r2_score
from Evalute_metrics import RMSE
import winsound
# 评估指标介绍https://www.cnblogs.com/mdevelopment/p/9456486.html
# EV: 解释回归模型的方差得分，[0,1]，接近1说明自变量越能解释因变量的方差变化
# MAE: 平均绝对误差，评估预测结果和真实数据集的接近程度的程度，越小越好
# MSE: 均方差， 计算拟合数据和原始数据对应样本点的误差的平方和的均值，越小越好
# R2: 判定系数，解释回归模型的方差得分，[0,1]，接近1说明自变量越能解释因变量的方差变化。
def model_compare():
    # =============================================================================
    # 模型选择
    model_br = BayesianRidge()  # 建立贝叶斯回归模型
    model_sgd = SGDRegressor() #梯度增强回归
    model_svr = SVR() # 建立支持向量回归模型 
    model_rm = RandomForestRegressor() # 建立普通线性回归模型
    #model_etc = ElasticNet() # 建立弹性网络回归模型
    model_mlp = MLPRegressor(
                              hidden_layer_sizes=200, 
                              max_iter=500,
                              random_state=7, #7
                              ) #神经网络回归
    model_gbr = GradientBoostingRegressor(
# =============================================================================
#                                            loss ='huber',
#                                            learning_rate = 0.01,
#                                            n_estimators = 200,
#                                            max_depth= 10

# =============================================================================
                                          ) # 建立梯度增强回归模型
    determined_params= {
# =============================================================================
#                      'n_estimators': 1200,
#                      'max_depth': 3,
#                      'subsample': 0.9,
#                      'min_child_weight': 0.001,
#                      'learning_rate': 0.1,
#                      'n_jobs': -1,
#                      'alpha':0,
#                      'lambda':1,
#                      'gamma':0.001,
# =============================================================================
                    }
    
    model_xgb = xgb.XGBRegressor(**determined_params) # XGBoost回归模型
    
    model_names = ['Bayes', 'SGD','SVR', 'RF',
                   'NFS-NN', 'XGBoost','BPNN']
    model_dir = [model_br, model_sgd, model_svr, model_rm, 
                 model_gbr, model_xgb, model_mlp]
    
    wrapper_dir = []
    for model in model_dir:
        wrapper_dir.append(MultiOutputRegressor(model))
        
    # 初始参数设置
    #virnum = list(range(30,60,5))
    virnum = [0,50,80,100,150]
    
    y_train_pre = []
    y_test_pre = []
    wrapper_train_metrics = []
    wrapper_test_metrics = []
    metrics_name = ['MAE','MSE','R2']
    df_train = pd.DataFrame(columns = metrics_name)
    df_test = pd.DataFrame(columns = metrics_name)
    # =============================================================================
    #样本数据导入
    X_train, X_test, y_train, y_test = dp.data_process(vir_num = 20,  
                                                       random_state = 42
                                                      )
    
    
    for i in range(len(virnum)):
        X_train, y_train = vs.VSG_ID(X_train,y_train,
                                                virtual= True, 
                                                affi=0.99, 
                                                vir_num = virnum[i], 
                                                method = 2, 
                                                seed1=42
                                                     )
        
    
        y_train_pre = [] # 各个模型预测的y值列表
        y_test_pre = [] # 创建测试集预测结果列表
    
        for wrapper in wrapper_dir:
            # 模型训练
            y_train_pre.append(wrapper.fit(X_train, y_train).predict(X_train))
            y_test_pre.append(wrapper.fit(X_train, y_train).predict(X_test))
            # 将训练模型的预测结果保存在列表中    
    # ============================================================================= 
        # 模型拟合与预测效果评估
        n_samples, n_features = X_train.shape # 总训练样本量，总特征量
        
        wrapper_metrics_name = [mean_absolute_error, 
                              mean_squared_error, r2_score]
        wrapper_train_metrics = [] # 回归训练评价指标列表
        wrapper_test_metrics = [] # 回归预测评价指标列表
        
    
        for j in range(len(wrapper_dir)):
            tmp_list = [] 
            tmp2_list = []
            for m in wrapper_metrics_name:
                tmp_score = m(y_train, y_train_pre[j]) # 计算每个回归指标结果
                tmp_list.append(tmp_score)
                tmp2_score = m(y_test, y_test_pre[j])
                tmp2_list.append(tmp2_score)
            
            wrapper_train_metrics.append(tmp_list)
            wrapper_test_metrics.append(tmp2_list)   
        
        df1 = pd.DataFrame(wrapper_train_metrics, index = model_names, 
                           columns = metrics_name) # 建立回归训练评估数据框
        df2 = pd.DataFrame(wrapper_test_metrics, index = model_names, 
                           columns = metrics_name) # 建立回归预测评估数据框
        
        df_train = pd.concat([df_train,df1])
        df_test = pd.concat([df_test,df2])
        
        
        print('\n\n') 
        print('Virtual numbers =  %d.\n' %(virnum[i]))
    
        print('samples: %d \t features: %d' %(n_samples, n_features)) 
        print(70*'-') # 打印分割线
        print('Regression train metrics:')
        print(df1)
        print(70*'-')
        print('Regression test metrics:')
        print(df2)
        print(70*'-')
         
    #winsound.Beep(3000, 440) #提示音
    return df_test,virnum,model_names,metrics_name

#防止程序在作为模块导入时被执行两遍
#https://zhuanlan.zhihu.com/p/21297237
if __name__ =='__main__':
    model_compare()