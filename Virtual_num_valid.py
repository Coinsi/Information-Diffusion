# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 12:05:31 2022

@author: Surface
验证虚拟样本生成数量、隶属度值对模型的提升效果
一般采用方法2效果更好
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import virtual_samples as vs
from sklearn.datasets import make_regression
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor #实现部分模型的多输出回归
from sklearn.metrics import explained_variance_score,mean_absolute_error,\
    mean_squared_error, mean_absolute_percentage_error, r2_score
import xgboost as xgb
import winsound
# =============================================================================
# 样本处理
#
metrics = [mean_absolute_error, mean_squared_error, r2_score]
metrics_name = ['MAE','MSE','R2']



virnum = list(range(0,50,10))
memb_values = list(np.arange(0.96,0.99,0.01))
# memb_values = [0.999]


#模型选择
determined_params= {
# =============================================================================
#                 'n_estimators': 400,
#                 'max_depth': 3,
#                 'subsample': 0.9,
#                 'min_child_weight': 0.001,
#                 'learning_rate': 0.1,
#                 'n_jobs': -1,
#                 'alpha':0,
#                 'lambda':1,
#                 'gamma':0.001,
# =============================================================================
                }
#model = xgb.XGBRegressor(**determined_params) # 建立梯度增强回归模型
#model = MLPRegressor(max_iter=400)
#model = SVR()
#model = RandomForestRegressor()
model = GradientBoostingRegressor(random_state=4)
wrapper =  MultiOutputRegressor(model)

result_train = []
result_test = []

for j in range(len(memb_values)):
    wrapper_train_metrics_arr = []
    wrapper_test_metrics_arr = []
    y_train_pre = []
    y_test_pre = []
    wrapper_train_metrics = []
    wrapper_test_metrics = []
    for i in range(len(virnum)):
        X_train, X_test, y_train, y_test = vs.VSG_ID(
                                                virtual= True, 
                                                affi=memb_values[j], 
                                                vir_num = virnum[i], 
                                                method = 2, 
                                                seed1=42
                                                     )
        # =============================================================================
    
        #模型训练
        y_train_pre.append(wrapper.fit(X_train, y_train).predict(X_train))
        y_test_pre.append(wrapper.fit(X_train, y_train).predict(X_test))
    
        
        tmp_list = [] 
        tmp2_list = []
        for m in metrics:
            tmp_score = m(y_train, y_train_pre[i]) # 计算每个回归指标结果
            tmp_list.append(tmp_score)
            tmp2_score = m(y_test, y_test_pre[i])
            tmp2_list.append(tmp2_score)
    
            
        wrapper_train_metrics.append(tmp_list)
        wrapper_test_metrics.append(tmp2_list)
    wrapper_train_metrics_arr = np.array(wrapper_train_metrics)       
    wrapper_test_metrics_arr = np.array(wrapper_test_metrics)       
    df1 = pd.DataFrame(wrapper_train_metrics_arr, index = virnum, 
                       columns = metrics_name) # 建立回归训练评估数据框
    df2 = pd.DataFrame(wrapper_test_metrics_arr, index = virnum, 
                       columns = metrics_name) # 建立回归预测评估数据框
    print(70*'=')
    print('Traning scores listed as:\n')
    print(df1)
    print(70*'=')
    print('Testing scores listed as:\n')
    print(df2)
    
    result_train.append(df1)
    result_test.append(df2)
# =============================================================================
# 可视化
plt.style.use('_mpl-gallery')
plt.figure(num = 'Fig5virnumchange', figsize = (16,5), edgecolor = 'k', frameon = True)
plot_list = {
            'marker':'o',
            'markersize': 5, 
            'linewidth': 2
            } # 统一设置标记点大小和线宽
font1 = {
         #'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14,
         }
ylimits = [(0,0.5),(0.025,0.5),(0.5,1)]
xlimits = [(0,50),(0,50),(0,50)]
for q in range(len(metrics)):
    plt.subplot(1,len(metrics),q+1)
    plt.plot(virnum,np.ones(len(virnum))*result_test[0].iloc[0,q],
             'k-',linewidth = 2, label = 'Without ID-DE')
    for p in range(len(result_test)):
        plt.plot(virnum,result_test[p].iloc[:,q],
                 label = 'Membership value = '+str(memb_values[p]),
                 **plot_list)
        #plt.fill_between(virnum, , alpha = 0.1)
    plt.legend(prop = font1)
    plt.xlabel('Samples',fontsize=18)
    #plt.ylabel('Samples',fontsize=18)
    plt.ylim(ylimits[q])
    plt.xlim(xlimits[q])
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.xlabel('Samples',fontsize=18)
    plt.title(metrics_name[q],fontsize=18)
# =============================================================================
# plt.figure(num = 'Predict', figsize = (16,12), edgecolor = 'k', frameon = True)
# for i in range(3):
#     plt.subplot(2,3,i+1)
#     plt.scatter(np.arange(1,y_test.shape[0]+1),y_test[:,i],'or')
#     plt.scatter(np.arange(1,y_test_pre.shape[0]+1),y_test_pre[:,i],'or')
# =============================================================================
  
winsound.Beep(3000, 440) #提示音
 
