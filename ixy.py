###################################################################
# File Name: ixy.py
# Author: Liukai
# mail: liukai@ncepu.edu.cn
# Created Time: 2020年12月13日 星期日 11时13分34秒
#=============================================================
#!/usr/bin/python

import numpy as np
import pandas as pd
import os 
import time
import pdb
from func import data_read
names = locals()
class Ixy():
    def __init__(self,data,label):
        self.data = data
        self.label = np.reshape(label,-1)

    def caculate_one(self,u1,u2,wind_size):
        cnts_x,bins_x = np.histogram(u1,wind_size)
        bins_x[-1]+=1
        pro_x = cnts_x/len(u1)
        cnts_y,bins_y = np.histogram(u2,wind_size)
        bins_y[-1]+=1
        pro_y = cnts_y/len(u2)
        pro_xy = np.ones([wind_size,wind_size])/len(u1)
        ixy = np.zeros([wind_size,wind_size])
        for i in range(wind_size):
            for j in range(wind_size):
                conditions_x = (bins_x[i]<=u1)&(u1<bins_x[i+1])
                conditions_y = (bins_y[j]<=u2)&(u2<bins_y[j+1])
                conditions = conditions_x*conditions_y
                pro_xy[i][j] *= len(conditions[conditions!=0])
                ixy[i][j]=pro_xy[i][j]*np.log2(pro_xy[i][j]/pro_x[i]/pro_y[j]+1e-10)
        sum_ixy = np.sum(ixy)
        return sum_ixy

    def caculate_dependent(self,wind_size):
        ixy = []
        for i in range(self.data.shape[1]):
            ixy.append(self.caculate_one(self.data[:,i],self.label,wind_size))
        return np.array(ixy)

    def caculate_independent(self,wind_size):
        nodes = self.data.shape[1]
        ixy = np.zeros([nodes,nodes])
        for i in range(nodes):
            for j in range(nodes):
                ixy[i][j] = self.caculate_one(self.data[:,i],self.data[:,j],wind_size)
        return ixy
def main():
    data,label = data_read('./train.xlsx')
    ixy = Ixy(data,label)
    #ixy = ixy.caculate_dependent(5)
    ixy = ixy.caculate_independent(5)
    ixy = np.nan_to_num(ixy)
    pd.DataFrame(ixy).to_excel('./adj.xlsx')
if __name__=='__main__':
    main()
    
