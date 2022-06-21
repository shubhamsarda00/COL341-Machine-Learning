# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 02:53:34 2021

@author: Asus
"""

import numpy as np
import pandas as pd
from numpy.linalg import norm      
import sys
from time import time

mode=sys.argv[1]
if(mode=='a'):

    train = pd.read_csv(sys.argv[2], index_col = 0)    
    test = pd.read_csv(sys.argv[3], index_col = 0)
    
    #y_train = np.array(train['Length of Stay'])
    y_train=pd.get_dummies(train['Length of Stay']).to_numpy()
    
    
    train = train.drop(columns = ['Length of Stay'])
    
    #Ensuring consistency of One-Hot Encoding
    
    data = pd.concat([train, test], ignore_index = True)
    cols = train.columns
    cols = cols[:-1]
    data = pd.get_dummies(data, columns=cols, drop_first=True)
    data = data.to_numpy()
    X_train = data[:train.shape[0], :]
    X_test = data[train.shape[0]:, :]
    
    #bias term
    X_train=np.concatenate((np.ones((X_train.shape[0],1)),X_train),axis=1)
    X_test=np.concatenate((np.ones((X_test.shape[0],1)),X_test),axis=1)
    
    def softmax(a):
        a=np.exp(a-np.max(a))
        return np.nan_to_num(a/(a.sum(axis=a.ndim-1).reshape(-1,1)),False)
    def CE_loss(y_true,y_preds):
        gamma=1e-15
        return -np.sum(np.multiply(y_true,np.log(np.clip(y_preds,gamma,1-gamma))))/y_true.shape[0]
        
  
    def grad_desc(num_iter,lr,X_train,y_train):
        W=np.zeros((X_test.shape[1],8))
        for i in range(num_iter):
            #print(softmax(np.matmul(X_train,W)))
            #print(i)
            #print(np.matmul(X_train,W))
            W-=lr/(y_train.shape[0])*(np.matmul(X_train.T,softmax(np.matmul(X_train,W))-y_train))
            #print(CE_loss(y_train,softmax(np.matmul(X_train,W))))
            #print((np.matmul(X_train.T,y_train-softmax(np.matmul(X_train,W)))).shape)
        return W
    
    def adaptive_grad_desc(num_iter,lr,X_train,y_train):
        W=np.zeros((X_test.shape[1],8))
        for i in range(num_iter):
            #print(softmax(np.matmul(X_train,W)))
            #print(i)
            #print(np.matmul(X_train,W))
            lr_=lr/((i+1)**.5)
            W-=lr_/(y_train.shape[0])*(np.matmul(X_train.T,softmax(np.matmul(X_train,W))-y_train))
            #print(CE_loss(y_train,softmax(np.matmul(X_train,W))))
            #print((np.matmul(X_train.T,y_train-softmax(np.matmul(X_train,W)))).shape)
        return W
    
    def backtrack_grad_desc(num_iter,lr,alpha,beta,X_train,y_train):
        W=np.zeros((X_test.shape[1],8))
        lr_=lr
        for i in range(num_iter):
            #if i%100==0:
            #    print(i)
            yh=softmax(np.matmul(X_train,W))
            dw=np.matmul(X_train.T,yh-y_train)/(y_train.shape[0])
            lr=lr_
            #print(yh)
            while(CE_loss(y_train,softmax(np.matmul(X_train,W-lr*dw)))-CE_loss(y_train,yh)>
                 -alpha*lr*(np.square(np.linalg.norm(dw)))):
                lr*=beta
                
                #print(CE_loss(y_train,softmax(np.matmul(X_train,W-lr/(y_train.shape[0])*dw)))-CE_loss(y_train,yh),
                 # -alpha*lr*(np.square(np.linalg.norm(dw))))
            
                #print(lr)
            #print(lr)
            W-=lr*dw
            #print((np.matmul(X_train.T,y_train-softmax(np.matmul(X_train,W)))).shape)
        return W
    
    f=open(sys.argv[4],'r')
    mode=f.readline().strip()
    if(mode=='1'):
        lr=float(f.readline().strip())
        num_iter=int(f.readline().strip())
        f.close()
        W=grad_desc(num_iter, lr, X_train, y_train)
        
        y_preds=np.squeeze(np.argmax(softmax(np.matmul(X_test,W)),axis=1)+1)
        
        f=open(sys.argv[5],'w')
        for y in y_preds:
            f.write(str(y)+'\n')
        f.close()
        
        f=open(sys.argv[6],'w')
        for w in W.flatten().squeeze():
            f.write(str(w)+'\n')
        f.close()
        
        
        
    elif mode=='2':
        lr=float(f.readline().strip())
        num_iter=int(f.readline().strip())
        f.close()
        W=adaptive_grad_desc(num_iter, lr, X_train, y_train)
        y_preds=np.squeeze(np.argmax(softmax(np.matmul(X_test,W)),axis=1)+1)
        
        f=open(sys.argv[5],'w')
        for y in y_preds:
            f.write(str(y)+'\n')
        f.close()
        
        f=open(sys.argv[6],'w')
        for w in W.flatten().squeeze():
            f.write(str(w)+'\n')
        f.close()
    
    else:
        lr,alpha,beta=list(map(float,f.readline().strip().split(',')))
        num_iter=int(f.readline().strip())
        f.close()
        W=backtrack_grad_desc(num_iter, lr, alpha, beta, X_train, y_train)
        
        y_preds=np.squeeze(np.argmax(softmax(np.matmul(X_test,W)),axis=1)+1)
        
        f=open(sys.argv[5],'w')
        for y in y_preds:
            f.write(str(y)+'\n')
        f.close()
        
        f=open(sys.argv[6],'w')
        for w in W.flatten().squeeze():
            f.write(str(w)+'\n')
        f.close()
        
elif(mode=='b'):
    train = pd.read_csv(sys.argv[2], index_col = 0)    
    test = pd.read_csv(sys.argv[3], index_col = 0)
    
    #y_train = np.array(train['Length of Stay'])
    y_train=pd.get_dummies(train['Length of Stay']).to_numpy()
    
    
    train = train.drop(columns = ['Length of Stay'])
    
    #Ensuring consistency of One-Hot Encoding
    
    data = pd.concat([train, test], ignore_index = True)
    cols = train.columns
    cols = cols[:-1]
    data = pd.get_dummies(data, columns=cols, drop_first=True)
    data = data.to_numpy()
    X_train = data[:train.shape[0], :]
    X_test = data[train.shape[0]:, :]
    
    #bias term
    X_train=np.concatenate((np.ones((X_train.shape[0],1)),X_train),axis=1)
    X_test=np.concatenate((np.ones((X_test.shape[0],1)),X_test),axis=1)
    
    def softmax(a):
        a=np.exp(a-np.max(a))
        return np.nan_to_num(a/(a.sum(axis=a.ndim-1).reshape(-1,1)),False)
    def CE_loss(y_true,y_preds):
        gamma=1e-15
        return -np.sum(np.multiply(y_true,np.log(np.clip(y_preds,gamma,1-gamma))))/y_true.shape[0]
        
    def grad_desc(bs,num_iter,lr,X_train,y_train):
        W=np.zeros((X_test.shape[1],8))
        for i in range(num_iter):
            for j in range(X_train.shape[0]//bs):
                #print(X_train[j*bs:(j+1)*bs,:].shape,y_train[j*bs:(j+1)*bs,:].shape)
                #print(softmax(np.matmul(X_train,W)))
                #print(i)
                #print(np.matmul(X_train,W))
                x,y=X_train[j*bs:(j+1)*bs,:],y_train[j*bs:(j+1)*bs,:]
                W-=lr/(bs)*(np.matmul(x.T,softmax(np.matmul(x,W))-y))
                #print(CE_loss(y_train,softmax(np.matmul(X_train,W))))
                #print((np.matmul(X_train.T,y_train-softmax(np.matmul(X_train,W)))).shape)
        return W

    def adaptive_grad_desc(bs,num_iter,lr,X_train,y_train):
        W=np.zeros((X_test.shape[1],8))
        for i in range(num_iter):
            for j in range(X_train.shape[0]//bs):
                #print(softmax(np.matmul(X_train,W)))
                #print(i)
                #print(np.matmul(X_train,W))
                lr_=lr/((i+1)**.5)
                x,y=X_train[j*bs:(j+1)*bs,:],y_train[j*bs:(j+1)*bs,:]
                W-=lr_/(bs)*(np.matmul(x.T,softmax(np.matmul(x,W))-y))
                #print(CE_loss(y_train,softmax(np.matmul(X_train,W))))
                #print((np.matmul(X_train.T,y_train-softmax(np.matmul(X_train,W)))).shape)
        return W

    def backtrack_grad_desc(bs,num_iter,lr,alpha,beta,X_train,y_train):
        
        W=np.zeros((X_test.shape[1],8))
        lr_=lr
        for i in range(num_iter):
        #    if(i%100==0):
         #       print(i)
            yh=softmax(np.matmul(X_train,W))
            dw=np.matmul(X_train.T,yh-y_train)/(y_train.shape[0])
            lr=lr_
            while(CE_loss(y_train,softmax(np.matmul(X_train,W-lr*dw)))-CE_loss(y_train,yh)>
                 -alpha*lr*(np.square(np.linalg.norm(dw)))):
                lr*=beta
            for j in range(X_train.shape[0]//bs):
                x,y=X_train[j*bs:(j+1)*bs,:],y_train[j*bs:(j+1)*bs,:]
                yh=softmax(np.matmul(x,W))
                dw=np.matmul(x.T,yh-y)/bs  
                
                #print(yh)
                #while(CE_loss(y,softmax(np.matmul(x,W-lr*dw)))-CE_loss(y,yh)>
                #     -alpha*lr*(np.square(np.linalg.norm(dw)))):
                 #   lr*=beta
                #print(lr)
                W-=lr*dw
            #print((np.matmul(X_train.T,y_train-softmax(np.matmul(X_train,W)))).shape)
        return W
    
    f=open(sys.argv[4],'r')
    mode=f.readline().strip()
    if(mode=='1'):
        lr=float(f.readline().strip())
        num_iter=int(f.readline().strip())
        bs=int(f.readline().strip())
        f.close()
        W=grad_desc(bs,num_iter, lr, X_train, y_train)
        
        y_preds=np.squeeze(np.argmax(softmax(np.matmul(X_test,W)),axis=1)+1)
        
        f=open(sys.argv[5],'w')
        for y in y_preds:
            f.write(str(y)+'\n')
        f.close()
        
        f=open(sys.argv[6],'w')
        for w in W.flatten().squeeze():
            f.write(str(w)+'\n')
        f.close()
        
        
        
    elif mode=='2':
        lr=float(f.readline().strip())
        num_iter=int(f.readline().strip())
        bs=int(f.readline().strip())
        f.close()
        W=adaptive_grad_desc(bs,num_iter, lr, X_train, y_train)
        y_preds=np.squeeze(np.argmax(softmax(np.matmul(X_test,W)),axis=1)+1)
        
        f=open(sys.argv[5],'w')
        for y in y_preds:
            f.write(str(y)+'\n')
        f.close()
        
        f=open(sys.argv[6],'w')
        for w in W.flatten().squeeze():
            f.write(str(w)+'\n')
        f.close()
    
    else:
        lr,alpha,beta=list(map(float,f.readline().strip().split(',')))
        num_iter=int(f.readline().strip())
        bs=int(f.readline().strip())
        f.close()
        W=backtrack_grad_desc(bs,num_iter, lr, alpha, beta, X_train, y_train)
        
        y_preds=np.squeeze(np.argmax(softmax(np.matmul(X_test,W)),axis=1)+1)
        
        f=open(sys.argv[5],'w')
        for y in y_preds:
            f.write(str(y)+'\n')
        f.close()
        
        f=open(sys.argv[6],'w')
        for w in W.flatten().squeeze():
            f.write(str(w)+'\n')
        f.close()

elif mode=='c':
    t=time()
    args=sys.argv
    
    train_path = args[2]
    test_path = args[3]
    train = pd.read_csv(train_path)    
    test = pd.read_csv(test_path)
    train=train.iloc[:,1:]
    test=test.iloc[:,1:]
    
    
    
    #y_train = np.array(train['Length of Stay'])
    y_train=pd.get_dummies(train['Length of Stay']).to_numpy()
    
    
    train = train.drop(columns = ['Length of Stay'])
    
    #Ensuring consistency of One-Hot Encoding
    
    data = pd.concat([train, test], ignore_index = True)
    cols = train.columns
    cols = cols[:-1]
    data = pd.get_dummies(data, columns=cols, drop_first=True)
    data.insert(0, 'Intercept', 1)
    data = data.to_numpy()
    X_train = data[:train.shape[0], :]
    X_test = data[train.shape[0]:, :]
    
    def softmax(a):
        m=np.max(a,axis=1).reshape(-1,1)
        #gamma=1e-15
        a=np.exp(a-m)
    #return a/(a.sum(axis=a.ndim-1).reshape(-1,1))
        return np.nan_to_num(a/(a.sum(axis=a.ndim-1).reshape(-1,1)),False)
    def CE_loss(y_true,y_preds):
        gamma=1e-15
        return -np.sum(np.multiply(y_true,np.log(np.clip(y_preds,gamma,1-gamma))))/y_true.shape[0]
    #np.log(np.clip(predictions, gamma, 1 - gamma)) , where gamma = 10^( -15) 
    def Kfold_CV(k,X):
        #np.random.shuffle(X)
       
        subsets=np.array_split(X,k)
        splits=[]
        for i in range(k):
            test=subsets[i]
            if i==0:
                train=np.vstack(subsets[i+1:])
                train=train.reshape((-1,train.shape[-1]))
            elif i==k-1:
                train=np.vstack(subsets[:i])
                train=train.reshape((-1,train.shape[-1]))
            else:    
                train=np.concatenate((np.vstack(subsets[:i]),np.vstack(subsets[i+1:])))
                train=train.reshape((-1,train.shape[-1]))
            x_test,y_test=test[:,:-1],test[:,-1]
            x_train,y_train=train[:,:-1],train[:,-1]
            splits.append((x_train,x_test,y_train,y_test))
        return splits
    
    def grad_desc(bs,num_iter,lr,X_train,y_train,t,X_test):
        t_hat=time()
        lmin=float('inf')
        
        Wmin=0
        W=np.zeros((X_train.shape[1],8))
        t=time()-t_hat
        i=0
        for i in range(num_iter):
           
            t_hat=time()
           
            for j in range(X_train.shape[0]//bs):
                #print(X_train[j*bs:(j+1)*bs,:].shape,y_train[j*bs:(j+1)*bs,:].shape)
                #print(softmax(np.matmul(X_train,W)))
                #print(i)
                #print(np.matmul(X_train,W))
                x,y=X_train[j*bs:(j+1)*bs,:],y_train[j*bs:(j+1)*bs,:]
                W-=lr/(bs)*(np.matmul(x.T,softmax(np.matmul(x,W))-y))
                #print(CE_loss(y_train,softmax(np.matmul(X_train,W))))
                #print((np.matmul(X_train.T,y_train-softmax(np.matmul(X_train,W)))).shape)
            l=CE_loss(y_train,softmax(np.matmul(X_train,W)))   
            
            if(l<lmin):
                
                lmin=l
        
                Wmin=W        
            t_hat=time()-t_hat
            t+=t_hat
            if(t>9*60):
            
                t_hat=time()
                #print(1,t,i,l)
                y_pred=softmax(np.matmul(X_test,Wmin))
                y_pred=np.squeeze(np.argmax(y_pred,axis=1)+1)
                f=open(args[4],'w')
                for y in y_pred:
                    f.write(str(y)+'\n')
                f.close()
                        
                f=open(args[5],'w')
                for w in Wmin.flatten().squeeze():
                    f.write(str(w)+'\n')
                f.close()        
                t+=time()-t_hat
        return Wmin
    
    
    
    def backtrack_grad_desc(bs,num_iter,lr,alpha,beta,X_train,y_train,t,X_test):
        t_counter=time()
        t_hat=time()     
        lr_=lr
    
        lmin=float('inf')
        Wmin=0
        W=np.zeros((X_train.shape[1],8))
        t+=time()-t_hat
        i=0
        for i in range(num_iter):
         
            t_hat=time()
            
        #    if(i%100==0):
         #       print(i)
            yh=softmax(np.matmul(X_train,W))
            dw=np.matmul(X_train.T,yh-y_train)/(y_train.shape[0])
            #lr=lr_
            #print(CE_loss(y_train,softmax(np.matmul(X_train,W-lr*dw)))-CE_loss(y_train,yh),-alpha*lr*(np.square(np.linalg.norm(dw))))
            while(CE_loss(y_train,softmax(np.matmul(X_train,W-lr*dw)))-CE_loss(y_train,yh)>
                 -alpha*lr*(np.square(np.linalg.norm(dw)))):
                lr*=beta
                
            for j in range(X_train.shape[0]//bs):
                x,y=X_train[j*bs:(j+1)*bs,:],y_train[j*bs:(j+1)*bs,:]
                yh=softmax(np.matmul(x,W))
                dw=np.matmul(x.T,yh-y)/bs  
    
                #print(yh)
                #while(CE_loss(y,softmax(np.matmul(x,W-lr*dw)))-CE_loss(y,yh)>
                #     -alpha*lr*(np.square(np.linalg.norm(dw)))):
                 #   lr*=beta
                #print(lr)
                W-=lr*dw
             
            
            l=CE_loss(y_train,softmax(np.matmul(X_train,W)))
            if(l<lmin):
           
                lmin=l
              
                Wmin=W 
            t_hat=time()-t_hat
            t+=t_hat
            
            if((time()-t_counter)>60 or t>9.2*60):
                t_counter=time()
                #print(1,t,i,l)
                t_hat=time()
                y_pred=softmax(np.matmul(X_test,Wmin))
                y_pred=np.squeeze(np.argmax(y_pred,axis=1)+1)
                f=open(args[4],'w')
                for y in y_pred:
                    f.write(str(y)+'\n')
                f.close()
                        
                f=open(args[5],'w')
                for w in Wmin.flatten().squeeze():
                   
                    f.write(str(w)+'\n')
                f.close()        
                t+=time()-t_hat    
        return Wmin
    
    
    
    bs=300
    t=time()-t
    #W=grad_desc(bs,500,1.265,X_train,y_train,t,X_test)
    W=backtrack_grad_desc(bs,750,2,.5,.9,X_train,y_train,t,X_test)
    
    y_pred=softmax(np.matmul(X_test,W))
    y_pred=np.squeeze(np.argmax(y_pred,axis=1)+1)
    
    f=open(args[4],'w')
    for y in y_pred:
        f.write(str(y)+'\n')
    f.close()
            
    f=open(args[5],'w')
    for w in W.flatten().squeeze():
        f.write(str(w)+'\n')
    f.close()


else:
   
    t=time()
    args=sys.argv
    def agg_cat(df,cat_col,num_col):
        
        a = df.groupby(cat_col)[num_col].agg(["std","max","mean"]).reset_index()
        a.columns = [cat_col] + ["grpby_"+ str(cat_col)+ "_" + str(a.columns[i]) +  "_" + str(num_col) for i in range(1,len(a.columns))]
        return df.merge(a,on = cat_col, how="left")

    #train_path = 'data/train.csv'
    #test_path = 'data/test.csv'
    train_path = sys.argv[2]
    test_path = sys.argv[3]    
    train_df = pd.read_csv(train_path, index_col = 0)    
    test_df = pd.read_csv(test_path, index_col = 0)
    train_df.insert(0, 'Intercept', 1)
    test_df.insert(0, 'Intercept', 1)
    
    #y_train = np.array(train['Length of Stay'])
    y_train=pd.get_dummies(train_df['Length of Stay']).to_numpy()
    
    
    train_df=train_df.drop(columns = ['Length of Stay'])
    
    
    # for c in train.columns:
    #     train[c] = (train[c]-train[c].mean())/train[c].std() 
    temp=None
    #Ensuring consistency of One-Hot Encoding
    temp = pd.concat([train_df, test_df], ignore_index = True)
    cols = train_df.columns[:-1]
    
    train_df1=pd.concat([train_df['APR Medical Surgical Description'].astype('str') + "_" + train_df['APR Severity of Illness Code'].astype('str'),
                    train_df['Age Group'].astype('str') + "_" + train_df['APR Severity of Illness Code'].astype('str'),
                    train_df["APR Risk of Mortality"].astype('str') + "_" + train_df['APR Severity of Illness Code'].astype('str'),
                    train_df['Emergency Department Indicator'].astype('str') + "_" + train_df['APR Severity of Illness Code'].astype('str'),
                    train_df['APR MDC Code'].astype('str') + "_" + train_df['APR Severity of Illness Code'].astype('str'),
                    train_df['Birth Weight'].astype('str') + "_" + train_df['APR Severity of Illness Code'].astype('str'),
                    train_df["Payment Typology 1"].astype('str') + "_" + train_df['APR Severity of Illness Code'].astype('str'),
                    train_df['Patient Disposition'].astype('str') + "_" + train_df['APR Severity of Illness Code'].astype('str'),
    #                 train_df['APR Medical Surgical Description'].astype('str') + "_" + train_df['Age Group'].astype('str'),
    #                 train_df["APR Risk of Mortality"].astype('str') + "_" + train_df['Age Group'].astype('str'),
    #                 train_df['Emergency Department Indicator'].astype('str') + "_" + train_df['Age Group'].astype('str'),
    #                 train_df['APR MDC Code'].astype('str') + "_" + train_df['Age Group'].astype('str'),
    #                 train_df['Birth Weight'].astype('str') + "_" + train_df['Age Group'].astype('str'),
    #                 train_df['Patient Disposition'].astype('str') + "_" + train_df['Age Group'].astype('str'),
    #                 train_df['Gender'].astype('str') + "_" + train_df['Age Group'].astype('str'),     
                     
                        train_df],axis=1,ignore_index=True)
    test_df1=pd.concat([test_df['APR Medical Surgical Description'].astype('str') + "_" + test_df['APR Severity of Illness Code'].astype('str'),
                    test_df['Age Group'].astype('str') + "_" + test_df['APR Severity of Illness Code'].astype('str'),
                    test_df["APR Risk of Mortality"].astype('str') + "_" + test_df['APR Severity of Illness Code'].astype('str'),
                    test_df['Emergency Department Indicator'].astype('str') + "_" + test_df['APR Severity of Illness Code'].astype('str'),
                    test_df['APR MDC Code'].astype('str') + "_" + test_df['APR Severity of Illness Code'].astype('str'),
                    test_df['Birth Weight'].astype('str') + "_" + test_df['APR Severity of Illness Code'].astype('str'),
                    test_df["Payment Typology 1"].astype('str') + "_" + test_df['APR Severity of Illness Code'].astype('str'),
                    test_df['Patient Disposition'].astype('str') + "_" + test_df['APR Severity of Illness Code'].astype('str'),
    #                 test_df['APR Medical Surgical Description'].astype('str') + "_" + test_df['Age Group'].astype('str'),
    #                 test_df["APR Risk of Mortality"].astype('str') + "_" + test_df['Age Group'].astype('str'),
    #                 test_df['Emergency Department Indicator'].astype('str') + "_" + test_df['Age Group'].astype('str'),
    #                 test_df['APR MDC Code'].astype('str') + "_" + test_df['Age Group'].astype('str'),
    #                 test_df['Birth Weight'].astype('str') + "_" + test_df['Age Group'].astype('str'),
    #                 test_df['Patient Disposition'].astype('str') + "_" + test_df['Age Group'].astype('str'),
    #                 test_df['Gender'].astype('str') + "_" + test_df['Age Group'].astype('str'),    
                       test_df],axis=1,ignore_index=True)
    
    temp = pd.concat([train_df1, test_df1], ignore_index = True)
    
    
    cols = train_df1.columns[:-1]
    
    
    # # cols = ['Health Service Area','Age Group','Gender','Type of Admission',
    # # "Patient Disposition","CCS Diagnosis Code","APR MDC Code",                     
    # # "APR Severity of Illness Code","APR Risk of Mortality","APR Medical Surgical Description","Birth Weight","Payment Typology 1"
    # # ,"Emergency Department Indicator"]
    
    
    
    temp = pd.get_dummies(temp, columns=cols, drop_first=True)
    test_temp = temp.iloc[train_df.shape[0]:, :]
    temp = temp.iloc[:train_df.shape[0], :]
    
    # temp=pd.concat([temp,
    #                agg_cat1(train_df,'APR Medical Surgical Description','APR Severity of Illness Code').iloc[:,-1:],
    #                agg_cat1(train_df,'Emergency Department Indicator','APR Severity of Illness Code').iloc[:,-1:],
    #                agg_cat1(train_df,'APR Severity of Illness Code','Age Group').iloc[:,-1:],
    #                agg_cat1(train_df,'APR Risk of Mortality','APR Severity of Illness Code').iloc[:,-1:],
    #                agg_cat1(train_df,'Age Group','APR Severity of Illness Code').iloc[:,-1:],
    #                agg_cat1(train_df,'APR DRG Code','APR Severity of Illness Code').iloc[:,-1:],
    #                agg_cat1(train_df,'APR MDC Code','APR Severity of Illness Code').iloc[:,-1:],
    #                agg_cat1(train_df,'CCS Procedure Code','APR Severity of Illness Code').iloc[:,-1:],
    #                agg_cat1(train_df,'CCS Diagnosis Code','APR Severity of Illness Code').iloc[:,-1:],
    #                agg_cat1(train_df,'Birth Weight','APR Severity of Illness Code').iloc[:,-1:],
    #                agg_cat1(train_df,'Patient Disposition','APR Severity of Illness Code').iloc[:,-1:],
    #                agg_cat1(train_df,'Gender','APR Severity of Illness Code').iloc[:,-1:],
    #                agg_cat1(train_df,'Payment Typology 1','APR Severity of Illness Code').iloc[:,-1:],
    #                agg_cat1(train_df,'Type of Admission','APR Severity of Illness Code').iloc[:,-1:],
    #                agg_cat1(train_df,'Payment Typology 3','APR Severity of Illness Code').iloc[:,-1:],
    #                agg_cat1(train_df,'Ethnicity','APR Severity of Illness Code').iloc[:,-1:],               
    #                     ],axis=1,ignore_index=True)
    
    # te=TargetEncoder(cols=train_df.columns[:-1],return_df=True)
    # temp=pd.concat([te.fit_transform(train_df,(np.argmax(y_train,axis=1).squeeze()+1)),temp],axis=1,ignore_index = True)
    # be=BaseNEncoder(cols=train_df.columns[:-1],return_df=True,base=2)
    # temp=pd.concat([temp,be.fit_transform(train_df,(np.argmax(y_train,axis=1).squeeze()+1))],axis=1,ignore_index=True)
    # cbe= CatBoostEncoder(cols=train_df.columns[:-1],return_df=True)
    # train_df=cbe.fit_transform(train_df,(np.argmax(y_train,axis=1).squeeze()+1))
    # bde=BackwardDifferenceEncoder(cols=train_df.columns[:-1],return_df=True)
    # temp=pd.concat([temp,bde.fit_transform(train_df)],axis=1,ignore_index=True)
    # counte=CountEncoder(cols=train_df.columns[:-1],return_df=True)
    # train_df=counte.fit_transform(train_df,(np.argmax(y_train,axis=1).squeeze()+1))
    # ge=GLMMEncoder(cols=train_df.columns[:-1],return_df=True)
    # temp=pd.concat([temp,ge.fit_transform(train_df,(np.argmax(y_train,axis=1).squeeze()+1))],axis=1,ignore_index=True)
    # he=HelmertEncoder(cols=train_df.columns[:-1],return_df=True)
    # temp=pd.concat([temp,he.fit_transform(train_df,(np.argmax(y_train,axis=1).squeeze()+1))],axis=1,ignore_index=True)
    
    # jse=JamesSteinEncoder(cols=train_df.columns[:-1],return_df=True)
    # train_df=jse.fit_transform(train_df,(np.argmax(y_train,axis=1).squeeze()+1))
    # pe=PolynomialEncoder(cols=train_df.columns[:-1],return_df=True)
    # train_df=pe.fit_transform(train_df,(np.argmax(y_train,axis=1).squeeze()+1))
    # se=SumEncoder(cols=train_df.columns[:-1],return_df=True)
    # temp=pd.concat([temp,se.fit_transform(train_df,(np.argmax(y_train,axis=1).squeeze()+1))],axis=1,ignore_index=True)
    #temp=se.fit_transform(temp,(np.argmax(y_train,axis=1).squeeze()+1))
    # woe=PolynomialWrapper(WOEEncoder(cols=train_df.columns[:-1],return_df=True))
    # train_df=pd.concat([temp,woe.fit_transform(train_df,(np.argmax(y_train,axis=1).squeeze()+1))],axis=1,ignore_index=True)
    
    temp=pd.concat([temp,
                   agg_cat(train_df,'APR Medical Surgical Description','Total Costs').iloc[:,-1:],
                   agg_cat(train_df,'Emergency Department Indicator','Total Costs').iloc[:,-1:],
                   agg_cat(train_df,'APR Severity of Illness Code','Total Costs').iloc[:,-1:],
                   agg_cat(train_df,'APR Risk of Mortality','Total Costs').iloc[:,-1:],
                   agg_cat(train_df,'Age Group','Total Costs').iloc[:,-1:],
                   agg_cat(train_df,'APR DRG Code','Total Costs').iloc[:,-1:],
                   agg_cat(train_df,'APR MDC Code','Total Costs').iloc[:,-1:],
                   agg_cat(train_df,'CCS Procedure Code','Total Costs').iloc[:,-1:],
                   agg_cat(train_df,'CCS Diagnosis Code','Total Costs').iloc[:,-1:],
                   agg_cat(train_df,'Birth Weight','Total Costs').iloc[:,-1:],
                   agg_cat(train_df,'Patient Disposition','Total Costs').iloc[:,-1:],
                   agg_cat(train_df,'Gender','Total Costs').iloc[:,-1:],
                   agg_cat(train_df,'Payment Typology 1','Total Costs').iloc[:,-1:],
                   agg_cat(train_df,'Type of Admission','Total Costs').iloc[:,-1:],
                   agg_cat(train_df,'Payment Typology 3','Total Costs').iloc[:,-1:],
                   agg_cat(train_df,'Ethnicity','Total Costs').iloc[:,-1:] ,
                   np.exp(train_df['Total Costs']),
                             np.square(train_df['Total Costs'])     
                       ],axis=1,ignore_index=True)
    
    test_temp=pd.concat([test_temp,
                   agg_cat(test_df,'APR Medical Surgical Description','Total Costs').iloc[:,-1:],
                   agg_cat(test_df,'Emergency Department Indicator','Total Costs').iloc[:,-1:],
                   agg_cat(test_df,'APR Severity of Illness Code','Total Costs').iloc[:,-1:],
                   agg_cat(test_df,'APR Risk of Mortality','Total Costs').iloc[:,-1:],
                   agg_cat(test_df,'Age Group','Total Costs').iloc[:,-1:],
                   agg_cat(test_df,'APR DRG Code','Total Costs').iloc[:,-1:],
                   agg_cat(test_df,'APR MDC Code','Total Costs').iloc[:,-1:],
                   agg_cat(test_df,'CCS Procedure Code','Total Costs').iloc[:,-1:],
                   agg_cat(test_df,'CCS Diagnosis Code','Total Costs').iloc[:,-1:],
                   agg_cat(test_df,'Birth Weight','Total Costs').iloc[:,-1:],
                   agg_cat(test_df,'Patient Disposition','Total Costs').iloc[:,-1:],
                   agg_cat(test_df,'Gender','Total Costs').iloc[:,-1:],
                   agg_cat(test_df,'Payment Typology 1','Total Costs').iloc[:,-1:],
                   agg_cat(test_df,'Type of Admission','Total Costs').iloc[:,-1:],
                   agg_cat(test_df,'Payment Typology 3','Total Costs').iloc[:,-1:],
                   agg_cat(test_df,'Ethnicity','Total Costs').iloc[:,-1:], 
                   np.exp(test_df['Total Costs']),
                             np.square(test_df['Total Costs'])     
                       ],axis=1,ignore_index=True)                
    
    # train_df=pd.concat([train_df,np.exp(train_df['Total Costs']),
    #                         np.square(train_df['Total Costs'])
    #                                   ],axis=1,ignore_index=True)
    
    s='''503, 2011, 306, 970, 305, 303, 265, 915, 189, 729, 2061, 676, 548, 0, 2071, 2072, 1979, 2079, 430, 34, 1162, 1977, 146, 2085, 2074, 2077, 1891, 2076, 48, 2073, 2075, 1976, 145, 29, 68, 1371, 1152, 1173, 1965, 1978, 353, 1859, 1964, 1986, 7, 47, 2081, 352, 3, 1151, 26, 1557, 1969, 32, 404, 64, 2, 25, 2068, 2070, 346, 1094, 85, 6, 1854, 403, 1554, 2078, 22, 2082, 1422, 1913, 83, 45, 1632, 4, 33, 436, 1174, 1968, 1163, 358, 44, 21, 124, 1642, 330, 437, 349, 19, 65, 1420, 15, 1630, 1343, 1168, 43, 1634, 1538, 40, 1907, 1347, 432, 433, 1980, 1905, 1995, 1627, 127, 1167, 1357, 431, 11, 123, 1916, 1262, 402, 91, 381, 1349, 1093, 1480, 1997, 82, 1159, 126, 1988, 421, 1985, 144, 2080, 1144, 1975, 1941, 380, 1914, 2083, 128, 131, 1339, 18, 1777, 1955, 2086, 1569, 348, 350, 46, 88, 1410, 122, 1812, 1736, 1345, 379, 2040, 2004, 2043, 240, 138, 1091, 361, 2069, 1981, 130, 1973, 1925, 1273, 2041, 1648, 84, 228, 125, 382, 1471, 2042, 333, 232, 136, 2007, 236, 1721, 1461, 351, 24, 2045, 39, 1926, 2039, 35, 223, 5, 23, 1092, 2044, 69, 90, 219, 28, 2038, 247, 1229, 1498, 1742, 1943, 1633, 243, 10, 120, 1379, 116, 435, 1261, 1999, 1574, 55, 139, 1713, 1649, 1998, 995, 806, 1953, 106, 1166, 1746, 1694, 1970, 60, 250, 1346, 2046, 17, 86, 54, 1729, 376, 1732, 1504, 1350, 1172, 1675, 362, 1268, 140, 360, 1165, 1146, 1341, 1472, 135, 1, 1954, 1863, 1302, 93, 1781, 42, 1960, 1637, 1949, 36, 1647, 1963, 137, 1956, 1962, 1950, 2047, 66, 1270, 8, 254, 401, 1564, 1337, 811, 1030, 1705, 620, 805, 1083, 142, 129, 1288, 14, 1489, 2048, 1636, 105, 2037, 56, 1958, 1470, 1714, 71, 79, 134, 1269, 216, 1495, 331, 397, 1537, 1200, 1967, 1291, 1701, 359, 20, 52, 372, 1049, 816, 1879, 1213, 1791, 121, 1486, 1712, 1833, 332, 1502, 1704, 1865, 258, 1335, 1655, 378, 49, 96, 1287, 1481, 1711, 212, 1555, 1424, 2036, 826, 898, 636, 1924, 1344, 117, 1359, 956, 876, 371, 1685, 952, 825, 635, 354, 41, 434, 624, 1577, 1846, 1546, 1787, 1423, 1813, 1148, 31, 769, 937, 584, 1390, 1217, 1877, 1776, 1375, 1355, 1670, 51, 78, 1063, 779, 594, 1850, 1836, 1989, 12, 1737, 1570, 836, 644, 958, 1398, 81, 1523, 1477, 1329, 1243, 1553, 1210, 1906, 1780, 1340, 1266, 1760, 1289, 2049, 261, 398, 53, 1727, 1629, 1490, 38, 623, 1571, 208, 629, 1591, 1438, 1719, 1735, 950, 849, 656, 1928, 1416, 1689, 1903, 498, 1856, 2035, 1391, 1644, 1256, 1216, 1388, 892, 821, 87, 1320, 1786, 1584, 420, 132, 342, 57, 1802, 400, 1638, 1386, 1659, 2028, 392, 1971, 347, 1319, 1282, 2084, 16, 1961, 1080, 818, 1286, 1209, 1404, 1761, 1652, 1316, 375, 1873, 1306, 1779, 972, 824, 2050, 1429, 2008, 1259, 1150, 1503, 1765, 1272, 269, 1457, 1959, 13, 1974, 1516, 1809, 338, 1587, 1626, 75, 922, 780, 1149, 1464, 1556, 70, 1697, 1664, 1211, 1170'''
    s=list(map(int,s.split(',')))

    #print(temp.shape,test_temp.shape)    
    X_train=temp.iloc[:,s].to_numpy()
    X_test=test_temp.iloc[:,s].to_numpy()
    #print(X_train.shape,X_test.shape)
    
    
    def softmax(a):
        m=np.max(a,axis=1).reshape(-1,1)
        #gamma=1e-15
        a=np.exp(a-m)
    #return a/(a.sum(axis=a.ndim-1).reshape(-1,1))
        return np.nan_to_num(a/(a.sum(axis=a.ndim-1).reshape(-1,1)),False)
    def CE_loss(y_true,y_preds):
        gamma=1e-15
        return -np.sum(np.multiply(y_true,np.log(np.clip(y_preds,gamma,1-gamma))))/y_true.shape[0]
    #np.log(np.clip(predictions, gamma, 1 - gamma)) , where gamma = 10^( -15) 
    def acc(y_true,y_preds):
        y_preds=np.argmax(y_preds,axis=1).squeeze()
        y_true=np.argmax(y_true,axis=1).squeeze()
        return np.count_nonzero(y_true-y_preds==0)/len(y_true)
    def Kfold_CV(k,X):
        #np.random.shuffle(X)
       
        subsets=np.array_split(X,k)
        splits=[]
        for i in range(k):
            test=subsets[i]
            if i==0:
                train=np.vstack(subsets[i+1:])
                train=train.reshape((-1,train.shape[-1]))
            elif i==k-1:
                train=np.vstack(subsets[:i])
                train=train.reshape((-1,train.shape[-1]))
            else:    
                train=np.concatenate((np.vstack(subsets[:i]),np.vstack(subsets[i+1:])))
                train=train.reshape((-1,train.shape[-1]))
            x_test,y_test=test[:,:-1],test[:,-1]
            x_train,y_train=train[:,:-1],train[:,-1]
            splits.append((x_train,x_test,y_train,y_test))
        return splits
    
    def grad_desc(bs,num_iter,lr,X_train,y_train,t,X_test):
        t_hat=time()
        lmin=float('inf')
        
        Wmin=0
        W=np.zeros((X_train.shape[1],8))
        t=time()-t_hat
        i=0
        for i in range(num_iter):
           
            t_hat=time()
           
            for j in range(X_train.shape[0]//bs):
                #print(X_train[j*bs:(j+1)*bs,:].shape,y_train[j*bs:(j+1)*bs,:].shape)
                #print(softmax(np.matmul(X_train,W)))
                #print(i)
                #print(np.matmul(X_train,W))
                x,y=X_train[j*bs:(j+1)*bs,:],y_train[j*bs:(j+1)*bs,:]
                W-=lr/(bs)*(np.matmul(x.T,softmax(np.matmul(x,W))-y))
                #print(CE_loss(y_train,softmax(np.matmul(X_train,W))))
                #print((np.matmul(X_train.T,y_train-softmax(np.matmul(X_train,W)))).shape)
            l=CE_loss(y_train,softmax(np.matmul(X_train,W)))   
            
            if(l<lmin):
                
                lmin=l
        
                Wmin=W        
            t_hat=time()-t_hat
            t+=t_hat
            if(t>14.2*60):
            
                t_hat=time()
                #print(1,t,i,l)
                y_pred=softmax(np.matmul(X_test,Wmin))
                y_pred=np.squeeze(np.argmax(y_pred,axis=1)+1)
                f=open(args[4],'w')
                for y in y_pred:
                    f.write(str(y)+'\n')
                f.close()
                        
                f=open(args[5],'w')
                for w in Wmin.flatten().squeeze():
                    f.write(str(w)+'\n')
                f.close()        
                t+=time()-t_hat
        return Wmin
    
    
    
   
    def backtrack_grad_desc(bs,num_iter,lr,alpha,beta,X_train,y_train,t,X_test):
        t_hat=time()   
        t_counter=time()
        lmin=float('inf')
        Wmin=0
        W=np.zeros((X_train.shape[1],8))
        t+=time()-t_hat
        for i in range(num_iter):
        #    if(i%100==0):
         #       print(i)
            t_hat=time()
            #print(i)
            yh=softmax(np.matmul(X_train,W))
            dw=np.matmul(X_train.T,yh-y_train)/(y_train.shape[0])
            
            while(CE_loss(y_train,softmax(np.matmul(X_train,W-lr*dw)))-CE_loss(y_train,yh)>
                 -alpha*lr*(np.square(np.linalg.norm(dw)))):
                lr*=beta
                
            
            for j in range(X_train.shape[0]//bs):
                x,y=X_train[j*bs:(j+1)*bs,:],y_train[j*bs:(j+1)*bs,:]
                yh=softmax(np.matmul(x,W))
                dw=np.matmul(x.T,yh-y)/bs  
                
                #print(yh)
                #while(CE_loss(y,softmax(np.matmul(x,W-lr*dw)))-CE_loss(y,yh)>
                #     -alpha*lr*(np.square(np.linalg.norm(dw)))):
                 #   lr*=beta
                #print(lr)
                W-=lr*dw
            
            l=CE_loss(y_train,softmax(np.matmul(X_train,W)))
            if(l<lmin):
           
                lmin=l
              
                Wmin=W 
            t_hat=time()-t_hat
            t+=t_hat
            if((time()-t_counter>60) or t>14.2*60):
                t_counter=time()
                t_hat=time()
                #print(t,i,l,lr)
                #y_pred=softmax(np.matmul(X_train,W))
                #print(i,CE_loss(y_train,y_pred),acc(y_train,y_pred))    
            
                
                y_pred=softmax(np.matmul(X_test,Wmin))
                y_pred=np.squeeze(np.argmax(y_pred,axis=1)+1)
                f=open(args[4],'w')
                for y in y_pred:
                    f.write(str(y)+'\n')
                f.close()
                        
                f=open(args[5],'w')
                for w in Wmin.flatten().squeeze():
                   
                    f.write(str(w)+'\n')
                f.close()        
                t+=time()-t_hat
            y_pred=softmax(np.matmul(X_train,W))
            #print(i,CE_loss(y_train,y_pred),acc(y_train,y_pred))    
            #print((np.matmul(X_train.T,y_train-softmax(np.matmul(X_train,W)))).shape)
        return W
    bs=350
    t=time()-t
    #W=grad_desc(bs,500,1.265,X_train,y_train,t,X_test)
    W=backtrack_grad_desc(bs,750,3,.5,.91,X_train,y_train,t,X_test)
    
    y_pred=softmax(np.matmul(X_test,W))
    y_pred=np.squeeze(np.argmax(y_pred,axis=1)+1)
    
    f=open(args[4],'w')
    for y in y_pred:
        f.write(str(y)+'\n')
    f.close()
            
    f=open(args[5],'w')
    for w in W.flatten().squeeze():
        f.write(str(w)+'\n')
    f.close()
    
        