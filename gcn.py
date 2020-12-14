###################################################################
# File Name: gcn.py
# Author: Liukai
# mail: liukai@ncepu.edu.cn
# Created Time: 2020年12月10日 星期四 10时20分16秒
#=============================================================
#!/usr/bin/python

import cv2
import numpy as np
import os 
import pdb
import tensorflow as tf

names = locals()

class Graph_convolution():
    def __init__(self,input_dim,output_dim,bias=True,activation='relu'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = tf.Variable(tf.random_normal([input_dim,output_dim]))
        self.bias = None
        self.act = activation
        if bias:
            self.bias=tf.Variable(tf.zeros(output_dim))
        if activation == 'relu':
            self.act = tf.nn.relu
        else:
            self.act = tf.nn.sigmoid
    
    
    def forward(self,x,adj):
        batch = x.shape[0].value
        nodes = x.shape[1].value
        out = tf.transpose(x,[1,0,2])
        out = tf.reshape(out,[nodes,-1])
        out = tf.matmul(adj,out)
        out = tf.reshape(out,[-1,nodes,self.input_dim])
        out = tf.transpose(out,[1,0,2])
        out = tf.reshape(out,[-1,self.input_dim])
        out = tf.matmul(out,self.weight)
        out = tf.reshape(out,[-1,nodes,self.output_dim])
        if self.bias:
            out +=self.bias
        return self.act(out)

    def __call__(self,x,adj):
         return self.forward(x,adj)

class Dense_layer():
    def __init__(self,input_dim,output_dim,activation=True,bias=True):
        self.weight = tf.Variable(tf.random_normal([input_dim,output_dim]))
        self.bias = None
        self.activation = activation
        if bias:
            self.bias=tf.Variable(tf.zeros(output_dim))

    def forward(self,x):
        out = tf.matmul(x,self.weight)
        if self.bias:
            out +=self.bias
        if self.activation:
            return tf.nn.sigmoid(out)
        else:
            return out

    def __call__(self,x):
        return self.forward(x)



class GCN():
    def __init__(self,input_dim,output_dim,nodes,gcn_net=[100],dense_net=[]):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nodes = nodes
        self.gcn_net = gcn_net
        self.dense_net = dense_net
        self.graph_layer = []
        self.dense_layer = []
        gcn_net = [input_dim]+self.gcn_net
        for i in range(len(gcn_net)-1):
            with tf.name_scope('Graph_variables{}'.format(i+1)):
                self.graph_layer.append(Graph_convolution(gcn_net[i],gcn_net[i+1]))
        dense_net = [int(gcn_net[-1]*self.nodes)]+self.dense_net

        for i in range(len(dense_net)-1):
            with tf.name_scope('Dense_variables{}'.format(i+1)):
                self.dense_layer.append(Dense_layer(dense_net[i],dense_net[i+1],True))
        self.output_layer=Dense_layer(dense_net[-1],self.output_dim,False)
    
    def forward(self,x,adj):
        out = x
        for i,layer in enumerate(self.graph_layer):
            with tf.name_scope('Graph_layer{}'.format(i+1)):
                out = layer(out,adj)
        out = tf.reshape(out,[-1,self.nodes*self.gcn_net[-1]])
        for i,layer in enumerate(self.dense_layer):
            with tf.name_scope('Dense_layer{}'.format(i+1)):
                out = layer(out)
        with tf.name_scope('Output_layer'):
            out = self.output_layer(out)
        return out

    def __call__(self,x,adj):
        return self.forward(x,adj)

def main():
    adj = tf.placeholder(tf.float32,[4,4])
    x = tf.placeholder(tf.float32,[4,100])
    model = GCN(100,1,4,[200,400,800,1600],[3200,1600,800,400,100])
    out = model(x,adj)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    logdir = './logdir'
    writer=tf.summary.FileWriter(logdir)
    writer.add_graph(sess.graph)


if __name__=='__main__':
    main()
    
