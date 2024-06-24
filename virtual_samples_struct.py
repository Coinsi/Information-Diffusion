# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 15:13:45 2022

@author: Surface
"""
import numpy as np
import random
import dataprocess as dp
import pandas as pd

class Samples():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.vir_num = 0
        self.affi = 0.999
        self.seed = 44
        self.__dim_x = self.X.shape[1]
        self.X_new = X
        self.y_new = y
        
    def vs_generate1(self, vir_num = 20, affi = 0.999, seed = 44):
        self.vir_num = vir_num
        self.affi = affi
        self.seed = seed
        data = np.concatenate([self.X,self.y],axis = 1)
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
        skew_u = 1-skew_l       #上偏度
        
        ##样本生成
        ##生成2倍的样本
        affi_matrix = self.affi*np.ones(np.shape(data))
        data_vir1_all = data-skew_l*np.sqrt(-2*var*np.log(affi_matrix))
        data_vir2_all = data+skew_u*np.sqrt(-2*var*np.log(affi_matrix))
        
        p = int(np.ceil(self.vir_num/(2*num_sample))-1)
        
        for i in range(p):       
            affi_matrix = affi_matrix-0.01
            temp_vir1 = data-skew_l*np.sqrt(-2*var*np.log(affi_matrix))
            temp_vir2 = data+skew_u*np.sqrt(-2*var*np.log(affi_matrix))
            data_vir1_all = np.concatenate((data_vir1_all,temp_vir1))
            data_vir2_all = np.concatenate((data_vir2_all,temp_vir2))
        
        ##从生成中的样本中选择vir_num个样本
        sample_list = [i for i in range(int(np.ceil(self.vir_num/2)))] #生成与样本数量一致的序列
        random_select1 = random.sample(sample_list,int(self.vir_num/2)) #随机从中选取vir_num个数
        random_select2 = random.sample(sample_list,self.vir_num-int(self.vir_num/2))
           
        data_vir1 = data_vir1_all[random_select1,:]
        data_vir2 = data_vir2_all[random_select2,:]
        data_new = np.concatenate((data_vir1,data,data_vir2))
      
        
        #数据打乱
        index = [i for i in range(len(data_new))]
        random.seed(self.seed1)
        random.shuffle(index)
        data_new = data_new[index]
        self.X_new = data_new[:,:self.__dim_x]
        self.y_new = data_new[:,self.__dim_x:]
        
        return self.X_new, self.y_new
    
    def vs_generate2(self, vir_num = 20, affi = 0.999, seed = 44):
        self.vir_num = vir_num
        self.affi = affi
        self.seed = seed
        
        data_ori = np.concatenate([self.X,self.y],axis = 1)
        data = data_ori
        dimention = data.shape[1]
            
        num_sample = data.shape[0]  # 生成后样本的数量
        num_total = [0]              # 记录样本生成的数量序列
        while num_total[-1] < self.vir_num+data_ori.shape[0]:
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
            affi_matrix = self.affi*np.ones(np.shape(data))
            data_vir1_all = data-skew_l*np.sqrt(-2*var*np.log(affi_matrix))
            data_vir2_all = data+skew_u*np.sqrt(-2*var*np.log(affi_matrix))
            data = np.concatenate((data,data_vir1_all,data_vir2_all))
                
            num_total.append(data.shape[0])
                
            # 从最后一次生成的数据中挑选
        sample_list = [i for i in range(num_total[-2]+1,num_total[-1])] # 确定最后一次生成样本的序号
        random_unselect = random.sample(sample_list,num_total[-1]-num_total[-2]-
                                           (self.vir_num+data_ori.shape[0]-num_total[-2])) # 从中随机选择提出的样本序号
        data_select = np.delete(data,random_unselect, axis = 0)           
                    
            
        index = [i for i in range(len(data_select))]
        random.shuffle(index)
        data_new = data_select[index]
            
        self.X_new = data_new[:,:self.__dim_x]
        self.y_new = data_new[:,self.__dim_x:]
            
        return self.X_new, self.y_new
        
    def vs_save(self):
        temp = np.concatenate((self.X_new,self.y_new), axis = 1)
        df = pd.DataFrame(temp)
        df.to_excel('New Samples.xlsx', sheet_name = 'biubiu')
        return df

if __name__ == '__main__':
    
    X_train, X_test, y_train, y_test = dp.data_process( 
                                                        random_state = 40
                                                        )
    
    data = Samples(X_train,y_train)
    b = data.X-data.X_new
    data.vs_generate2(60,0.99)
    vir_num = range(20,10,100)
    data.vs_generate2(vir_num)
    #data.vs_save()
        
        