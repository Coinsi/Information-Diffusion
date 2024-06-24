# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 10:23:32 2022

@author: Yang, Zeyuan
基于信息扩散原理（ID）生成虚拟样本（VS）
样本数据,Table2.xlsx,Table3.xlsx,其中,
Table2.xlsx为输入，特征数=12，
Table3.xlsx为输出，输出维度=8.

该模块需要以dataprocess数据处理模块为基础

VSG_ID()需要设置的参数为：
1. 是否采用样本增强方法（True,False)
2. 初始隶属度值（一般越接近于1越好，但不能等于1）
3. 生成样本的数量
4. 生成样本采用的方法（1：变换隶属度值；2：循坏迭代，一般而言，方法2效果更好）
5. 样本打乱及挑选的随机数
注：一般需要调整的参数主要是生成样本的数量以及选用的隶属度值的大小
"""
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split

def VSG_ID(X,y,
          virtual= True, 
          affi=0.99, 
          vir_num = 20, 
          method = 2, 
          seed1=42
          ):
    
#设置虚拟样本生成数量及隶属度值


    # 根据训练集生成虚拟样本
    if virtual: 
        dim_x = X.shape[1] # 特征的数量
        
        #采用方法1生成虚拟样本：变换隶属度值
        if method == 1:
            data = np.concatenate([X,y],axis = 1)
            num_sample = data.shape[0]
            dimention = data.shape[1]
            
            #求解样本均值与方差
            mean = np.mean(data,0)
            var = np.var(data,axis = 0)
            
            
            #求解数据的偏度
            num_l = [] #统计每一列中小于均值的样本的数量
            for i in range(dimention):
                num_l.append(np.sum(data[:,i]<mean[i]))
            N_l = np.array(num_l)
            skew_l = N_l/num_sample #下偏度
            skew_u = 1-skew_l  #上偏度
            
            ##样本生成
            ##生成2倍的样本
            affi_matrix = affi*np.ones(np.shape(data))
            data_vir1_all = data-skew_l*np.sqrt(-2*var*np.log(affi_matrix))
            data_vir2_all = data+skew_u*np.sqrt(-2*var*np.log(affi_matrix))
            
        
            p = int(np.ceil(vir_num/(2*num_sample))-1)
        
            for i in range(p):       
                affi_matrix = affi_matrix-0.01
                temp_vir1 = data-skew_l*np.sqrt(-2*var*np.log(affi_matrix))
                temp_vir2 = data+skew_u*np.sqrt(-2*var*np.log(affi_matrix))
                data_vir1_all = np.concatenate((data_vir1_all,temp_vir1))
                data_vir2_all = np.concatenate((data_vir2_all,temp_vir2))
            
            ##从生成中的样本中选择vir_num个样本
            sample_list = [i for i in range(int(np.ceil(vir_num/2)))] #生成与样本数量一致的序列
            random_select1 = random.sample(sample_list,int(vir_num/2)) #随机从中选取vir_num个数
            random_select2 = random.sample(sample_list,vir_num-int(vir_num/2))
               
            data_vir1 = data_vir1_all[random_select1,:]
            data_vir2 = data_vir2_all[random_select2,:]
            data_new = np.concatenate((data_vir1,data,data_vir2))
          
            
            #数据打乱
            index = [i for i in range(len(data_new))]
            random.seed(seed1)
            random.shuffle(index)
            data_new = data_new[index]
            X = data_new[:,:dim_x]
            y = data_new[:,dim_x:]
            
        #采用方法2生成虚拟样本：反复迭代输入
        else:
            data_ori = np.concatenate([X,y],axis = 1)
            data = data_ori
            dimention = data.shape[1]
            
            num_sample = data.shape[0]   # 生成后样本的数量
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
                #data_vir1_all = data-skew_l*np.sqrt(-2*var*np.log(affi_matrix))
                #data_vir2_all = data+skew_u*np.sqrt(-2*var*np.log(affi_matrix))
                
                data_vir_all = data+skew_u*np.sqrt(-2*var*np.log(affi_matrix))
                
                ## 删除不满足约束条件的行
                #data_vir1_all = data_vir1_all[np.all(data_vir1_all > 0, axis=1)]
                #data_vir2_all = data_vir2_all[np.all(data_vir2_all > 0, axis=1)]
                
                #data = np.concatenate((data,data_vir1_all,data_vir2_all))
                data = np.concatenate((data,data_vir_all))
                
                
                num_total.append(data.shape[0])
                
            # 从最后一次生成的数据中挑选
            sample_list = [i for i in range(num_total[-2]+1,num_total[-1])] # 确定最后一次生成样本的序号
            random_unselect = random.sample(sample_list,num_total[-1]-num_total[-2]-
                                            (vir_num+data_ori.shape[0]-num_total[-2])) # 从中随机选择提出的样本序号
            data_select = np.delete(data,random_unselect, axis = 0)           
                    
            
            index = [i for i in range(len(data_select))]
            random.shuffle(index)
            data_new = data_select[index]
            
            X = data_new[:,:dim_x]
            y = data_new[:,dim_x:]
            
    return X, y

if __name__ =='__main__':
   T2 = pd.read_excel('Table2.xlsx')
   T3 = pd.read_excel('Table3.xlsx')
   
   data_X = T2.iloc[2:,1:]
   data_Y = T3.iloc[2:,1:5]
   data = pd.concat([data_X, data_Y], axis=1)
   data = data.astype(float)
   data = data.reset_index(drop = True)
   data.columns = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12',
                   'y1','y2','y3','y4']
   df_data = data
   data = df_data.values
   dim_x = 12
   X, y = VSG_ID(data[:,:dim_x],data[:,dim_x:],
             virtual= True, 
             affi=0.999, 
             vir_num = 11, 
             method = 2, 
             seed1=32
             )
   temp = np.concatenate((X,y), axis = 1)
   df = pd.DataFrame(temp)
   # df.to_excel('New Samples.xlsx', sheet_name = 'biubiu')