# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 18:37:32 2021

@author: Asus
"""

import numpy as np
import pandas as pd
import sys

inp=sys.argv[1]
output=sys.argv[2]
param=sys.argv[3]
data=pd.read_csv(inp+'train_data_shuffled.csv',header=None)
test=pd.read_csv(inp+'public_test.csv',header=None)
X_test=test.iloc[:,:-1].to_numpy()/255.
X,y=data.iloc[:,:-1].to_numpy()/255.,pd.get_dummies(data.iloc[:,-1:],columns=data.columns[-1:]).to_numpy()
#print(X.shape,y.shape,X_test.shape)


f=open(param)
epoch=int(f.readline())
bs=int(f.readline())
arc=list(map(int,f.readline().strip()[1:-1].split(',')))
#print(arc)
adaptive=False
if(int(f.readline()))==1:
    adaptive=True
lr=float(f.readline())
af=int(f.readline())
l=int(f.readline())
np.random.seed(int(f.readline()))
f.close()
def softmax(a):
    m=np.max(a,axis=1).reshape(-1,1)
    #gamma=1e-15
    a=np.exp(a-m)
    #return a/(a.sum(axis=a.ndim-1).reshape(-1,1))
    return np.nan_to_num(a/(a.sum(axis=a.ndim-1).reshape(-1,1)),False)
def relu(a):
    return np.where(a>0,a,0)
def drelu(a):
    return np.where(a>0,1,0)
def tanh(a):
    return np.tanh(a)  
def dtanh(a):
    return 1-np.square(a)
def sigmoid(a):
    return 1/(1+np.exp(-a))
    #return expit(a)
def dsig(a):
    #return sigmoid(a)*(1-sigmoid(a))
    return a*(1-a)
    #return (1-expit(a))*expit(a)
def CE_loss(y_true,y_preds):
    gamma=1e-15
    return -np.sum(np.multiply(y_true,np.log(np.clip(y_preds,gamma,1-gamma))))/y_true.shape[0]

def mse(y_true,y_preds):  
    return np.sum(np.square(y_true.reshape(-1,1)-y_preds.reshape(-1,1)))/y_preds.shape[0]


def acc(y_true,y_preds):
    y_preds=np.argmax(y_preds,axis=1).squeeze()
    y_true=np.argmax(y_true,axis=1).squeeze()
    return np.count_nonzero(y_true-y_preds==0)/len(y_true)

class ANN:
    hidden_units=daf=af=l=W=Z=0
    
    def __init__(self,hidden_units,af,num_features,loss):
        
        self.hidden_units=hidden_units
        self.l=loss
        if(af==0):
            self.af=sigmoid
            self.daf=dsig
        elif(af==1):
            self.af=tanh
            self.daf=dtanh
        else:
            self.af=relu
            self.daf=drelu
        if(loss==0):
            self.loss=CE_loss
        else:
            self.loss=mse 
        self.W=[]
        self.b=[]
        temp=np.float32(np.random.normal(0,1,(num_features+1,hidden_units[0])))*(np.sqrt(2/(num_features+1+hidden_units[0])))
        self.W.append(temp[1:,:])
        self.b.append(temp[0])
        for i in range(1,len(hidden_units)):
            temp=np.float32(np.random.normal(0,1,(hidden_units[i-1]+1,hidden_units[i])))*(np.sqrt(2/(hidden_units[i-1]+1+hidden_units[i])))
            self.W.append(temp[1:,:])
            self.b.append(temp[0])
        self.Z=[]    
    def fp(self,X):
        self.Z=[]
        
        self.Z.append(X)
        
#         for i in range(len(self.W)):
#             self.Z.append(self.af(np.matmul(self.Z[i],self.W[i])+self.b[i]))
            
#         if(self.l==0):
#             return softmax(self.Z[-1]) 
        for i in range(len(self.W)-1):
            self.Z.append(self.af(np.matmul(self.Z[i],self.W[i])+self.b[i]))
            
        if(self.l==0):
            #print(self.Z[-1].shape,self.W[-1].shape,self.b[-1].shape)
            self.Z.append(np.matmul(self.Z[-1],self.W[-1])+self.b[-1])
            return softmax(self.Z[-1])
            
        else:
            self.Z.append(self.af(np.matmul(self.Z[-1],self.W[-1])+self.b[-1]))
        return self.Z[-1]
    def bp(self,X,y,lr):
        yh=self.fp(X)
        g=yh-y
        #print(g)
        for i in reversed(range(len(self.W))):
            
            deriv_af=self.daf(self.Z[i+1])
            
            if(i<len(self.W)-1 or self.l==1):
                
                g=np.multiply(g,deriv_af)
            dw=np.matmul(self.Z[i].T,g)/yh.shape[0]
            #print(dw[0])
            #db=np.matmul(dummy,g)/yh.shape[0]
            db=np.sum(g,axis=0)/yh.shape[0]
            
            g=np.dot(g,self.W[i].T)
            self.W[i]-=dw*lr
            self.b[i]-=db*lr
#     def bp(self,X,y,lr):
#         yh=self.fp(X)
#         g=yh-y
#         #print(g)
#         dummy=np.ones(yh.shape[0])
#         for i in reversed(range(len(self.W))):
#             deriv_af=self.daf(self.Z[i+1])
#             g=np.multiply(deriv_af,g)
#             dw=np.matmul(self.Z[i].T,g)/yh.shape[0]
#             #print(dw[0])
#             #db=np.matmul(dummy,g)/yh.shape[0]
#             db=np.sum(g,axis=0)/yh.shape[0]
            
#             g=np.dot(g,self.W[i].T)
#             self.W[i]-=dw*lr
#             self.b[i]-=db*lr
    
    def fit(self,bs,epochs,X,y,lr,adaptive=False):
        for i in range(len(self.W)):
            self.W[i]=np.float64(self.W[i])
            self.b[i]=np.float64(self.b[i])
        l=[]    
        for i in range(epochs):
            if(adaptive):
                lr/=(i+1)**.5
            for j in range(X.shape[0]//bs):
#                 if(j==5 and i==0):
#                     for k in range(len(a.W)):
#                         temp=np.concatenate((a.b[k].reshape(1,-1),a.W[k]),axis=0)
#                         np.save('essentials/part_a_and_b/multiclass_dataset/tc_3/w_'+str(k+1)+'_iter.npy',temp)
#                         print(np.max(np.load('essentials/part_a_and_b/multiclass_dataset/tc_3/ac_w_'+str(k+1)+'_iter.npy')-temp))
                    
                self.bp(X[j*bs:(j+1)*bs,:],y[j*bs:(j+1)*bs,:],lr)
            #print(i,l[-1],acc(y,self.fp(X)))
            #print(i,self.loss(y_test,y_pred),acc(y_test,y_pred))
        return l    
    
    def pred(self,X):
        return self.fp(X)

a=ANN(arc,af,1024,l)
a.fit(bs,epoch,X,y,lr,adaptive)

for i in range(len(a.W)):
    temp=np.concatenate((a.b[i].reshape(1,-1),a.W[i]),axis=0)
    np.save(output+'w_'+str(i+1)+'.npy',temp)

y_h=a.fp(X)
np.save(output+'predictions.npy',np.argmax(y_h,axis=1).squeeze())
