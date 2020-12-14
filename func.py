import pandas as pd
import numpy as np
import pdb
import os

def rm_rf(path):
    file_list = os.listdir(path)
    for name in file_list:
        os.remove(path+'/'+name)
    os.removedirs(path)




def data_read(path):#数据集读取
    data = pd.read_excel(path)
    data = np.array(data.values)
    label = np.reshape(data[:,0],(-1,1))
    #data = data[:,:9]#设置变量个数
    #data = data[:,1:]
    return data,label

def normalize(data,label,mode='max_min'):#归一化
    def max_min(dat):
        ma = np.max(dat,axis = 0)
        mi = np.min(dat,axis = 0)
        dat_=[]
        mo = ma-mi+1e-10
        for line in dat:
            dat_.append((line-mi)/mo)
        return np.array(dat_)
    def standard(dat):
        mean = np.mean(dat,axis = 0)
        std = np.std(dat,axis = 0)+1e-10
        dat_ = (dat-mean)/std
        pdb.set_trace()
        return dat_
    return max_min(data),max_min(label)

def un_normalize(predict,label,mode='max_min'):#反归一化
    def un_max_min(label,predict):
        ma = np.max(label,axis = 0)
        mi = np.min(label,axis =0)
        dat_ = []
        mo = ma-mi
        for line in predict:
            dat_.append(line*mo+mi)
        return np.array(dat_)
    return un_max_min(label,predict)



def analysis(predict , label):
    def get_RMSE(predict_,label_):
        data = np.sum(pow((predict_-label_),2))
        data_RMSE = np.sqrt(1/len(predict_)*data)
        return data_RMSE
    def get_MAE(predict_,label_):
        data = np.sum(abs(predict_-label_))
        data_MAE = 1/len(predict_)*data
        return data_MAE
    def get_MSE(predict_,label_):
        data = np.sum(pow((predict_-label_),2))
        data_MSE = 1/len(predict_)*data
        return data_MSE
    def get_MAPE(predict_,label_):
        data = np.sum(abs(predict_-label_)/abs(label_))
        data_MAPE = 1/len(predict_)*data
        return data_MAPE
    def get_Acc(predict_,label_):
        Acc = 0
        for i in range(len(predict_)):
            if(abs(predict_[i]-label_[i])/label[i]<0.05):
                Acc+=1
        return Acc/len(predict_)
    def get_R2(predict_,label_):
        mean = np.mean(predict_)
        SSres = np.sum(pow((predict_-mean),2))
        SStot = np.sum(pow((predict_-label_),2))
        return 1-SStot/SSres
    def get_R(predict_,label_):
        cov = np.cov(predict_,label_)
        std_p = np.var(predict_)
        std_l = np.var(label_)
        return cov[0,1]/pow(std_p*std_l,0.5)
    predict = np.reshape(predict,(-1))
    label = np.reshape(label,(-1))
    dic = {'RMSE':get_RMSE(predict,label),'MSE':get_MSE(predict,label),'MAE':get_MAE(predict,label),'MAPE':get_MAPE(predict,label),'Acc':get_Acc(predict,label),'R2':get_R2(predict,label),'R':get_R(predict,label)}
    return dic 


if __name__ == '__main__':
    predict,label = data_read('data_output.xlsx')
    label = np.reshape(predict[:,2],(-1))
    predict = np.reshape(predict[:,1],(-1))
    

    pdb.set_trace()
    
    #dic = analysis(predict,label)
    #saver = pd.DataFrame([dic])
    #saver.to_excel('analy.xlsx')
    #print(dic)
