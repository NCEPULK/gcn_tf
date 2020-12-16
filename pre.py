###################################################################
# File Name: pre.py
# Author: Liukai
# mail: liukai@ncepu.edu.cn
# Created Time: 2020年12月11日 星期五 18时55分27秒
#=============================================================
#!/usr/bin/python

import cv2
import numpy as np
import os 
import pdb
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import pandas as pd
from ixy import Ixy
from gcn import GCN
from func import data_read,normalize,un_normalize,analysis
names = locals()
class Config():
    def __init__(self):
        self.epoch = 3000  #leanring epoch
        self.lr = 0.0001     #learning rate
        self.ts = 3        #time step
        self.bs = 30         #batch size
        self.train_path = 'train.xlsx'
        self.test_path  = 'test.xlsx'
        self.adj_path = 'adj.xlsx'
        self.nodes = 27     #graph nodes
        self.gl = [12,24] #graph layers
        self.dl = [24]   #dense layers


    def show(self):
        s

def compute_loss(pre,lab):
    return tf.reduce_mean(tf.square(pre-lab))
def get_batch(data,label,batch,time_step,mode='train'):
    ma = len(label)%(batch)
    length = len(label)//(batch)
    if mode =='train':
        start = np.random.randint(ma)
        batch_data = np.reshape(data[start:start+length*batch],[length,batch,time_step,-1])
        batch_label = np.reshape(label[start:start+length*batch],[length,batch,1])
    if mode == 'test':
        batch_data = []
        batch_label= []
        for i in range(len(label)-time_step):
            batch_data.append(data[i:i+time_step])
            batch_label.append(label[i:i+time_step])
    return np.array(batch_data),np.array(batch_label)
def trans_gcn(data,label,time_step):
    x=[]
    y=[]
    for i in range(len(data)-time_step):
        x.append(data[i:i+time_step])
        y.append(label[i+time_step])
    return np.array(x),np.array(y)
def get_adj(path='adj.xlsx'):
    adj = pd.read_excel(path)
    return adj.values[:,1:]

def train(config,model_name='-1'):
    data,label = data_read(config.train_path)
    adjacency = get_adj(config.adj_path) 
    x,y = normalize(data,label)
    x = x[:,:config.nodes]
    x,y = trans_gcn(x,y,config.ts)
    adj = adjacency[:config.nodes,:config.nodes]
    ph_adj = tf.placeholder(tf.float32,[config.nodes,config.nodes],'adj')
    ph_data = tf.placeholder(tf.float32,[None,config.nodes,config.ts],'data')
    ph_label = tf.placeholder(tf.float32,[None,1],'label')
    model = GCN(config.ts,1,config.nodes,config.gl,config.dl)
    out = model(ph_data,ph_adj)
    loss_op = compute_loss(out,ph_label)
    tf.summary.scalar('loss',loss_op)
    train_op = tf.train.AdamOptimizer(config.lr).minimize(loss_op)
    merge_op = tf.summary.merge_all()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./logdir/'+model_name)
    writer.add_graph(sess.graph)
    saver = tf.train.Saver()
    _loss = []
    for i in range(config.epoch):
        batch_data,batch_label=get_batch(x,y,config.bs,config.ts)
        for j in range(len(batch_label)):
            _data = batch_data[j].transpose([0,2,1])
            _,loss,summary = sess.run([train_op,loss_op,merge_op],feed_dict={ph_adj:adj,ph_data:_data,ph_label:batch_label[j]})
            _loss.append(loss)
        writer.add_summary(summary,i)
        if i%100==0:
            print('epoch====={}\t\t\tloss======{}\n\n'.format(i,np.mean(_loss)))
            _loss=[]
    saver.save(sess,'./model/'+model_name+'/best_model.ckpt')

            
def predict(config,model_name='-1'):
    data,label = data_read(config.train_path)
    adjacency = get_adj(config.adj_path)
    x,y = normalize(data,label)
    x = x[:,:config.nodes]
    adj = adjacency[:config.nodes,:config.nodes]
    ph_adj = tf.placeholder(tf.float32,[config.nodes,config.nodes],'adj')
    ph_data = tf.placeholder(tf.float32,[None,config.nodes,config.ts],'data')
    ph_label = tf.placeholder(tf.float32,[None,1],'label')
    model = GCN(config.ts,1,config.nodes,config.gl,config.dl)
    out = model(ph_data,ph_adj)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess,'./model/'+model_name+'/best_model.ckpt')
    batch_data,batch_label = get_batch(x,y,config.bs,config.ts,'test')
    _data = batch_data.transpose([0,2,1])
    out=sess.run(out,feed_dict={ph_adj:adj,ph_data:_data,ph_label:batch_label[0]})
    pre = un_normalize(out,label)
    ana = analysis(pre,label[config.ts:])
    save_data = pd.DataFrame({'pre':np.reshape(pre,-1),'lab':np.reshape(label[config.ts:],-1)})
    save_ana  = pd.DataFrame(ana,index=[0])
    save_data.to_excel('./output/4/train_data_output.xlsx')
    save_ana.to_excel('./output/4/train_data_analysis.xlsx')
    #axis = list(range(len(pre)))
    plt.plot(pre,'r')
    plt.plot(label[config.ts:],'g')
    plt.show()
    



def main():
    TIME = time.strftime('%y-%m-%d-%H-%M-%S',time.localtime())
    #model_name = TIME
    model_name='3'
    print(model_name)
    config = Config()
    train(config,model_name)
    tf.reset_default_graph()
    #predict(config,model_name)

if __name__=='__main__':
    main()
    
