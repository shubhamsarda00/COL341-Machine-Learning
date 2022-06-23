# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 18:46:28 2021

@author: Asus
"""
from time import time
th=time()

import numpy as np
import pandas as pd
import sys
np.random.seed(146)

inp=sys.argv[1]
output=sys.argv[2]
param=sys.argv[3]
f=open(param)
arc=list(map(int,f.readline().strip()[1:-1].split(',')))
f.close()


    

data=pd.read_csv(inp+'train_data_shuffled.csv',header=None)

X,y=data.iloc[:,:-1].to_numpy()/255.,pd.get_dummies(data.iloc[:,-1:],columns=data.columns[-1:]).to_numpy()
#print(X.shape,y.shape)

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
        self.final_W=self.W
        self.final_b=self.b
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
    def bp_mom(self,X,y,lr,wt,bt,gamma=.9):
        yh=self.fp(X)
        g=yh-y
        #print(g)
        #print(gamma)
        
        
        for i in reversed(range(len(self.W))):
            
            deriv_af=self.daf(self.Z[i+1])
            if(i<len(self.W)-1 or self.l==1):
                g=np.multiply(g,deriv_af)
            dw=np.matmul(self.Z[i].T,g)/yh.shape[0]
            #db=np.matmul(dummy,g)/yh.shape[0]
            db=np.sum(g,axis=0)/yh.shape[0]
            g=np.dot(g,self.W[i].T)
            #print(gamma*wt[i])
            change=gamma*wt[i]+dw*lr
            self.W[i]-=change
            wt[i]=change
            #print(wt[i])
            change=gamma*bt[i]+db*lr
            self.b[i]-=change
            bt[i]=change
            
    
    def bp_rmsprop(self,X,y,lr,wt,bt,gamma=.9):
        
        yh=self.fp(X)
        g=yh-y
        
        #print(g)
        #print(gamma)
       
        e=1e-8
        for i in reversed(range(len(self.W))):
            
            deriv_af=self.daf(self.Z[i+1])
            
            if(i<len(self.W)-1 or self.l==1):
                
                g=np.multiply(g,deriv_af)
            dw=np.matmul(self.Z[i].T,g)/yh.shape[0]
            #db=np.matmul(dummy,g)/yh.shape[0]
            db=np.sum(g,axis=0)/yh.shape[0]
            g=np.dot(g,self.W[i].T)
            #print(gamma*wt[i])
            temp=gamma*wt[i]+0.1*(dw**2)
            self.W[i]-=lr*dw/(np.sqrt(temp+e))
            wt[i]=temp
            #print(wt[i])
            temp=gamma*bt[i]+.1*(db**2)
            self.b[i]-=lr*db/(np.sqrt(temp+e))
            bt[i]=temp
    def bp_adam(self,X,y,lr,vwt,vbt,mwt,mbt,b1=.9,b2=.999):
        yh=self.fp(X)
        g=yh-y
        #print(g)
        #print(gamma)
       
        e=1e-8
        b1t=1
        b2t=1
        for i in reversed(range(len(self.W))):
            
            deriv_af=self.daf(self.Z[i+1])
            
            if(i<len(self.W)-1 or self.l==1):
                
                g=np.multiply(g,deriv_af)
            dw=np.matmul(self.Z[i].T,g)/yh.shape[0]
            #db=np.matmul(dummy,g)/yh.shape[0]
            db=np.sum(g,axis=0)/yh.shape[0]
            g=np.dot(g,self.W[i].T)
            #print(gamma*wt[i])
            vwt[i]=b2*vwt[i]+(1-b2)*(dw**2)
            vbt[i]=b2*vbt[i]+(1-b2)*(db**2)
            mwt[i]=b1*mwt[i]+(1-b1)*(dw)
            mbt[i]=b1*mbt[i]+(1-b1)*(db)
            #print(mbt[i])
            b1t*=b1
            b2t*=b2
            self.W[i]-=lr*(mwt[i]/(1-b1t))/(np.sqrt(vwt[i]/(1-b2t)+e))
            self.b[i]-=lr*(mbt[i]/(1-b1t))/(np.sqrt(vbt[i]/(1-b2t)+e))
            
            
    def bp_nadam(self,X,y,lr,vwt,vbt,mwt,mbt,b1=.9,b2=.999):
        yh=self.fp(X)
        g=yh-y
        #print(g)
        #print(gamma)
       
        e=1e-8
        b1t=1
        b2t=1
        for i in reversed(range(len(self.W))):
            
            deriv_af=self.daf(self.Z[i+1])
            
            if(i<len(self.W)-1 or self.l==1):
                
                g=np.multiply(g,deriv_af)
            dw=np.matmul(self.Z[i].T,g)/yh.shape[0]
            #db=np.matmul(dummy,g)/yh.shape[0]
            db=np.sum(g,axis=0)/yh.shape[0]
            g=np.dot(g,self.W[i].T)
            #print(gamma*wt[i])
            vwt[i]=b2*vwt[i]+(1-b2)*(dw**2)
            vbt[i]=b2*vbt[i]+(1-b2)*(db**2)
            mwt[i]=b1*mwt[i]+(1-b1)*(dw)
            mbt[i]=b1*mbt[i]+(1-b1)*(db)
            #print(mbt[i])
            b1t*=b1
            b2t*=b2
            self.W[i]-=lr*(b1*mwt[i]/(1-b1t)+(1-b1)*dw/(1-b1t))/(np.sqrt(vwt[i]/(1-b2t)+e))
            self.b[i]-=lr*(b1*mbt[i]/(1-b1t)+(1-b1)*db/(1-b1t))/(np.sqrt(vbt[i]/(1-b2t)+e))
    
    def bp_nag(self,X,y,lr,wt,bt,gamma=.9):
        W_temp=self.W
        b_temp=self.b
        for i in range(len(self.W)):
            self.W[i]-=gamma*wt[i]
            self.b[i]-=gamma*bt[i]
            
        yh=self.fp(X)
        g=yh-y
        
        #print(g)
        #print(gamma)
        
        for i in reversed(range(len(self.W))):
            
            deriv_af=self.daf(self.Z[i+1])
            
            if(i<len(self.W)-1 or self.l==1):
                
                g=np.multiply(g,deriv_af)
            dw=np.matmul(self.Z[i].T,g)/yh.shape[0]
            #db=np.matmul(dummy,g)/yh.shape[0]
            db=np.sum(g,axis=0)/yh.shape[0]
               
            g=np.dot(g,self.W[i].T)
           
            temp=gamma*wt[i]+lr*dw
            
            W_temp[i]-=temp
            wt[i]=temp
            #print(wt[i])
            temp=gamma*bt[i]+lr*db
            b_temp[i]-=temp
            bt[i]=temp
        self.W=W_temp
        self.b=b_temp
    
    def fit(self,bs,epochs,X,y,lr,t,adaptive=False):
        t_counter=time()
        th=time()
        for i in range(len(self.W)):
            
            
            self.W[i]=np.float64(self.W[i])
            self.b[i]=np.float64(self.b[i])
#         l=[]
        lmin=float('inf')
        t+=time()-th
        for i in range(epochs):
            th=time()
            if(adaptive):
                lr/=(i+1)**.5
            for j in range(X.shape[0]//bs):
                th=time()
#                 if(j==5 and i==0):
#                     for k in range(len(a.W)):
#                         temp=np.concatenate((a.b[k].reshape(1,-1),a.W[k]),axis=0)
#                         np.save('essentials/part_a_and_b/multiclass_dataset/tc_3/w_'+str(k+1)+'_iter.npy',temp)
#                         print(np.max(np.load('essentials/part_a_and_b/multiclass_dataset/tc_3/ac_w_'+str(k+1)+'_iter.npy')-temp))
                    
                self.bp(X[j*bs:(j+1)*bs,:],y[j*bs:(j+1)*bs,:],lr)
                t+=time()-th
                th=time()
                if(4.8*60>t>4.7*60):
                    l=self.loss(y,self.fp(X))
                    if(l<lmin):
                        lmin=l
                        self.final_W=self.W
                        self.final_b=self.b
                        for k in range(len(self.final_W)):
                            temp=np.concatenate((self.final_b[k].reshape(1,-1),self.final_W[k]),axis=0)
                            np.save(output+'w_'+str(k+1)+'.npy',temp)
                        #print(str(t)+': saved:'+str(lmin))
                        
                t+=time()-th
            
            
            th=time()    
            l=self.loss(y,self.fp(X))
            if(l<lmin):
                lmin=l
                self.final_W=self.W
                self.final_b=self.b
            t+=time()-th
            th=time()
            if(time()-t_counter>30 or 290>t>4.4*60):
                t_counter=time()
                for k in range(len(self.final_W)):
                    temp=np.concatenate((self.final_b[k].reshape(1,-1),self.final_W[k]),axis=0)
                    np.save(output+'w_'+str(k+1)+'.npy',temp)
                #print(str(t)+': saved:'+str(lmin))
            t+=time()-th
           
            if(t>300):
                return lmin
#             y_pred=self.fp(X_test)
#             l.append((self.loss(y,self.fp(X)),acc(y,self.fp(X))))
#             print(i,l[-1])
#             print(i,self.loss(y_test,y_pred),acc(y_test,y_pred))
        return lmin    
    def fit_mom(self,bs,epochs,X,y,lr,t,adaptive=False):
        t_counter=time()
        th=time()
        for i in range(len(self.W)):
            
            
            self.W[i]=np.float64(self.W[i])
            self.b[i]=np.float64(self.b[i])
#         l=[]
        lmin=float('inf')
       
        wt=[0]*len(self.W)
        bt=[0]*len(self.W)
        t+=time()-th
        for i in range(epochs):
            th=time()
            if(adaptive):
                lr/=(i+1)**.5
            for j in range(X.shape[0]//bs):
                th=time()
                self.bp_mom(X[j*bs:(j+1)*bs,:],y[j*bs:(j+1)*bs,:],lr,wt,bt)
                t+=time()-th
                th=time()
                if(4.8*60>t>4.7*60):
                    l=self.loss(y,self.fp(X))
                    if(l<lmin):
                        lmin=l
                        self.final_W=self.W
                        self.final_b=self.b
                        for k in range(len(self.final_W)):
                            temp=np.concatenate((self.final_b[k].reshape(1,-1),self.final_W[k]),axis=0)
                            np.save(output+'w_'+str(k+1)+'.npy',temp)
                        #print(str(t)+': saved:'+str(lmin))
                        
                t+=time()-th
            th=time()    
            l=self.loss(y,self.fp(X))
            if(l<lmin):
                lmin=l
                self.final_W=self.W
                self.final_b=self.b
            t+=time()-th
            th=time()
            if(time()-t_counter>30 or 290>t>4.4*60):
                t_counter=time()
                for k in range(len(self.final_W)):
                    temp=np.concatenate((self.final_b[k].reshape(1,-1),self.final_W[k]),axis=0)
                    np.save(output+'w_'+str(k+1)+'.npy',temp)
                #print(str(t)+': saved:'+str(lmin))
            t+=time()-th
            if(t>300):
                return lmin
#             y_pred=self.fp(X_test)
#             l.append((self.loss(y,self.fp(X)),acc(y,self.fp(X))))
#             print(i,l[-1])
#             print(i,self.loss(y_test,y_pred),acc(y_test,y_pred))
        return lmin
        
    def pred(self,X):
        return self.fp(X)
        
    
a=0    
if(len(arc)==2):
    f=open(output+'my_params.txt','w')
    f.write('150')
    f.write('\n')
    f.write('50')
    f.write('\n')
    f.write('0')
    f.write('\n')
    f.write('0.4')
    f.write('\n')
    f.write('2')
    f.write('\n')
    f.write('0')
    f.write('\n')
    f.write('0')
    f.write('\n')
    f.write('146')
    f.close()
    #print(2)
    a=ANN([256,46],2,1024,0)
    t=time()-th
    loss=a.fit(50,150,X,y,.4,t)

else:
    f=open(output+'my_params.txt','w')
    f.write('150')
    f.write('\n')
    f.write('150')
    f.write('\n')
    f.write('0')
    f.write('\n')
    f.write('0.01')
    f.write('\n')
    f.write('2')
    f.write('\n')
    f.write('0')
    f.write('\n')
    f.write('1')
    f.write('\n')
    f.write('146')
    #print(4)
    f.close()
    a=ANN([512,256,128,64,46],2,1024,0)    
    t=time()-th
    a.fit_mom(150,150,X,y,.01,t)        