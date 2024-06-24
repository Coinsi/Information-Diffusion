# -*- coding: utf-8 -*-
"""
Created on Fri May  6 21:20:59 2022

@author: Yang Zeyuan
数据预处理，包括从excel中提取数据、(数据标准化)、数据打乱、训练测试划分
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import virtual_samples as vs


# 提取excel数据
def data_extract_xls():
    T2 = pd.read_excel('Table2.xlsx')
    T3 = pd.read_excel('Table3.xlsx')
    
    data_X = T2.iloc[2:,1:]
    data_Y = T3.iloc[2:,1:5]
    data = pd.concat([data_X, data_Y], axis=1)
    data = data.astype(float)
    data = data.reset_index(drop = True)
    data.columns = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12',
                    'y1','y2','y3','y4']
    
    return data


# =============================================================================
# 数据打乱与划分
def data_process(vir_num = 0, random_state = 42):
    '''
       指定进行数据打乱与划分的原始数据集
    '''
    test_size=0.3 # 测试集、训练集数量划分
    dim_x = 12    # X的特征数量
    
    df_data = data_extract_xls()
    data = df_data.values
    
    ## 生成虚拟样本作为真实样本
    X,y = vs.VSG_ID(data[:,:dim_x],data[:,dim_x:],
                    virtual= True, 
                    affi=0.99, 
                    vir_num = vir_num, 
                    method = 2, 
                    seed1=42
                    )
    
    X_train, X_test, y_train, y_test = train_test_split(
         X,y,test_size = test_size, random_state = random_state)
    

    return X_train, X_test, y_train, y_test

   
if __name__ =='__main__':
    X_train, X_test, y_train, y_test = data_process(vir_num = 1000,  
                                                    random_state = 42
                                                    )
    temp = np.concatenate((X_train,y_train), axis = 1)
    temp2 = np.concatenate((X_test,y_test), axis = 1)
    temp3 = np.concatenate((temp,temp2),axis = 0)
    df = pd.DataFrame(temp3)
    # df.to_excel('New Samples.xlsx', sheet_name = 'biubiu')
