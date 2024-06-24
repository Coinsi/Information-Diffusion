# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 22:14:33 2022

@author: Surface
承接model_compares.py的结果可视化
ID-DE对不同学习模型的提升效果直方图
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from  model_compares import model_compare
'''
该函数用于比较不同模型采用虚拟样本后的精度提升
输出为训练集和测试集的评估结果
需要调整的参数有：虚拟样本序列，采用生成虚拟样本的方法method
'''
df_test,virnum,model_names,metrics_name = model_compare()

metrics_br = df_test.loc['Bayes']
metrics_sgd = df_test.loc['SGD']
metrics_svr = df_test.loc['SVR']
metrics_rm = df_test.loc['RF']
metrics_gbr = df_test.loc['NFS-NN']
metrics_xgb = df_test.loc['XGBoost']
metrics_mlp = df_test.loc['BPNN']

test_dir = [metrics_br,metrics_sgd,metrics_svr,
            metrics_rm,metrics_gbr,metrics_xgb,metrics_mlp]

# 修改index
for p in test_dir:
    p.index = virnum

metrics_vir_mean = []
metrics_vir_var = []
metrics_original = []
metrics_virual = []
for i,metrics in enumerate(test_dir):
    metrics_original.append(metrics.iloc[0,:]) 
    metrics_virual.append(metrics.iloc[1:,:])
    metrics_vir_mean.append(metrics_virual[i].mean(axis = 0))
    metrics_vir_var.append(metrics_virual[i].std(axis = 0))
    
    #metrics.plot(subplots = True)
    #plt.tight_layout()

metrics_vir_mean = pd.concat(metrics_vir_mean,axis = 1)
metrics_vir_var  = pd.concat(metrics_vir_var,axis = 1)
metrics_original = pd.concat(metrics_original,axis = 1)
metrics_vir_mean.columns = model_names
metrics_vir_var.columns = model_names
metrics_original.columns = model_names
metrics_impr = metrics_vir_mean-metrics_original  #平均提升效果
# =============================================================================
# 是否采用虚拟样本对不同学习模型的提升效果直方图对比

x = np.arange(metrics_original.values.shape[1])
total_width,n = 0.8,2
error_params=dict(elinewidth=4,ecolor='r',capsize=5)#设置误差标记参数
bar_width = total_width/n
font1 = {
         #'family': 'Times New Roman',
         'weight': 'normal',
         'size': 16,
         }
ylimits = [(0.15,0.45),(0.025,0.2),(0.1,1)]
for j in range(7,len(df_test),7):
    plt.style.use('_mpl-gallery')
    plt.figure(figsize = (16,6), edgecolor = 'k', frameon = True)
    for i in range(metrics_original.values.shape[0]):
        plt.subplot(1,3,i+1)
        plt.title(metrics_name[i],fontsize=18)
        plt.bar(x,metrics_original.values[i],bar_width,
                label = 'With MTD-VSG')
        plt.bar(x+bar_width, df_test.values[j:j+len(model_names),i],bar_width,
                #yerr = metrics_vir_var.values[i],error_kw = error_params,
                label = 'Without MTD-VSG')
        plt.xticks(x+bar_width/2,model_names,rotation = 70)#设置x轴的标签
        #plt.grid(axis='y',ls='-',color='r',alpha=0.3)
        plt.legend(prop = font1)
        plt.ylim(ylimits[i])
        plt.tick_params(labelsize=16)
        plt.tight_layout()