# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 13:05:53 2022

@author: Surface
样本生成方法调试，该方法是通过反复迭代输入进而生成无穷数量的虚拟样本
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import dataprocess as dp

#设置虚拟样本生成数量及隶属度值
affi=0.99
vir_num = 50
method = 2
seed1=42
seed2=42
view = True


#导入原始样本并进行训练集、测试集划分
X_train, X_test, y_train, y_test = dp.data_process( 
                                                    random_state = seed1
                                                    )
temp1 = np.concatenate([X_train,X_test],axis = 0);
temp2 = np.concatenate([y_train,y_test],axis = 0);
data_ori = np.concatenate([temp1,temp2],axis = 1);
  
dim_x = X_train.shape[1]
#data_ori = np.concatenate([X_train,y_train],axis = 1)
data = data_ori
dimention = data.shape[1]

num_sample = data.shape[0]  # 生成后样本的数量
num_total = [0]              # 记录样本生成的数量序列
while num_total[-1] < vir_num+data_ori.shape[0]:
    #求解样本均值与方差
    mean = np.mean(data, 0)
    var = np.var(data, axis = 0)
    
    #求解数据的偏度
    num_l = [] #统计每一列中小于均值的样本的数量
    for i in range(dimention):
        num_l.append(np.sum(data[:,i] < mean[i]))
        
    N_l = np.array(num_l)
    skew_l = N_l/num_sample #下偏度
    skew_u = 1-skew_l       #上偏度
    
    ##样本生成
    affi_matrix = affi*np.ones(np.shape(data))
    data_vir1_all = data-skew_l*np.sqrt(-2*var*np.log(affi_matrix))
    data_vir2_all = data+skew_u*np.sqrt(-2*var*np.log(affi_matrix))
    
    #data_vir_all = data+np.sqrt(-2*var*np.log(affi_matrix))
    
    ## 删除不满足约束条件的行
    #data_vir1_all = data_vir1_all[np.all(data_vir1_all >= 0, axis=1)]
    #data_vir2_all = data_vir2_all[np.all(data_vir2_all >= 0, axis=1)]
    
    data = np.concatenate((data,data_vir1_all,data_vir2_all))
    #data = np.concatenate((data,data_vir_all))
    
    num_total.append(data.shape[0])
    
# 从最后一次生成的数据中挑选
sample_list = [i for i in range(num_total[-2]+1,num_total[-1])] # 确定最后一次生成样本的序号
random_unselect = random.sample(sample_list,num_total[-1]-num_total[-2]-
                                (vir_num+data_ori.shape[0]-num_total[-2])) # 从中随机选择提出的样本序号
data_select = np.delete(data,random_unselect, axis = 0)

## 可视化原始样本与新样本
if view:
    fig1 = plt.figure(1,figsize = [16,9])
    ax = []
    for i in range(dimention):
        ax.append(fig1.add_subplot(5,4,i+1))

        plt.scatter(data_ori[:,i],data_ori[:,i],
                    color = 'r',marker = 'o',label = 'original sample')

        plt.scatter(data_select[data_ori.shape[0]+1:,i],data_select[data_ori.shape[0]+1:,i],
                    color = 'c',marker = '*',
                    label = 'virtual sample')
        plt.legend()
        plt.tight_layout()
        plt.show()
        

index = [i for i in range(len(data_select))]
random.shuffle(index)
data_new = data_select[index]

X_train = data_new[:,:dim_x]
y_train = data_new[:,dim_x:]