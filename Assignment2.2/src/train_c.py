# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 11:12:52 2021

@author: Asus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import pandas as pd
from time import time
import sys

class cifar10_dataset(Dataset):    
    def __init__(self,data,train = True,img_transform=None):
        self.img_transform = img_transform
        self.is_train = train   
#         data = pd.read_csv(data_csv, header=None)
        if self.is_train:
            self.images,self.labels=data[0],data[1]            
#             images = data.iloc[:,1:].to_numpy()
#             labels = data.iloc[:,0].astype(int)
        else:
            self.images=data
#             images = data.iloc[:,:]
#             labels = None  
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,i):
        image = self.images[i]
        #image = np.array(image).astype(np.uint8).reshape((32, 32, 3),order='F')
        if self.is_train:
            label = self.labels[i]
        else:
            label = -1
        image = self.img_transform(image)
        return image,label
    
X_train=pd.read_csv(sys.argv[1],header=None)
X_test=pd.read_csv(sys.argv[2],header=None)
X_train,y_train=np.array(X_train.iloc[:,1:],dtype=np.uint8),np.array(X_train.iloc[:,0],dtype=int)
X_test,y_test=np.array(X_test.iloc[:,1:],dtype=np.uint8),np.array(X_test.iloc[:,0],dtype=int)
X_train,X_test=X_train.reshape((-1,3,32,32)).transpose(0,2,3,1),X_test.reshape((-1,3,32,32)).transpose(0,2,3,1)
X_train.shape,y_train.shape,X_test.shape,y_test.shape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(cnn,train_data,epochs,opt,loss,test_data,n,t):

    th=time()
#     losses=[]
#     accuracies=[]
    a=0
    amax=0
    t+=time()-th
    for i in range(epochs):
        th=time()
        l=0
        for j,(X,y) in enumerate(train_data):
                
            #print(X.shape,y.shape,type(X),type(y))
            yh=cnn(X.to(device))
            train_loss=loss(yh,y.type(torch.LongTensor).to(device))
            opt.zero_grad()
            l+=train_loss.item()
            train_loss.backward()
            opt.step()
        t+=time()-th
        th=time()
        
        with torch.no_grad():
#             losses.append(l)
            l/=len(train_data) 
            a=0
            for k,(X,y) in enumerate(test_data):
                y_preds=cnn(X.to(device))
                y_preds=torch.argmax(y_preds,dim=1).squeeze()
                a+=(y.to(device)==y_preds).sum().item()
            a/=n
#             ac=0
#             for k,(X,y) in enumerate(test):
#                 y_preds=cnn(X)
#                 y_preds=torch.argmax(y_preds,dim=1).squeeze()
#                 ac+=(y==y_preds).sum().item()
#             ac/=n    
#            accuracies.append()
            t+=time()-th
            th=time()
            print("Epoch: "+str(i+1)+" Train loss: "+str(l)+" test accuracy: " +str(a)+" time: "+str(t))  
            if(a>amax):
                amax=a
                torch.save(cnn.state_dict(),sys.argv[3])
            t+=time()-th
        
            
    return amax


class fused_block(nn.Module):
    def __init__(self,i):
        super(fused_block,self).__init__()
        self.c1=nn.Conv2d(i,4*i,3,padding='same')
        self.bn1=nn.BatchNorm2d(4*i)
        self.c2=nn.Conv2d(4*i,i,1,padding='same')
        self.bn2=nn.BatchNorm2d(i)
    def __call__(self,X):
        return F.relu(self.bn2(self.c2(F.relu(self.bn1(self.c1(X))))))
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.c1=nn.Conv2d(3,64,3,padding='same')
        self.bn1=nn.BatchNorm2d(64)
        self.fb1=fused_block(64)
        self.p1=nn.MaxPool2d(2,2)
        self.c2=nn.Conv2d(64,128,3,padding='same')
        self.bn2=nn.BatchNorm2d(128)
        self.fb2=fused_block(128)
        self.p2=nn.MaxPool2d(2,2)
        self.c3=nn.Conv2d(128,256,3,padding='same')
        self.bn3=nn.BatchNorm2d(256)
        self.p3=nn.MaxPool2d(2,2)
        self.p4=nn.MaxPool2d(2,2)
        self.c4=nn.Conv2d(256,512,3,padding='same')
        self.bn4=nn.BatchNorm2d(512)
        self.fc0=nn.Linear(2048,256)
        self.fc1=nn.Linear(256,128)
        self.fc2=nn.Linear(128,10)
    def forward(self,X):
        X=F.relu(self.bn1(self.c1(X)))
        X=self.p1(self.fb1(X))
        X=F.relu(self.bn2(self.c2(X)))
        X=self.p2(self.fb2(X))
        X=self.p4(F.relu(self.bn3(self.c3(X))))
        X=self.p4(F.relu(self.bn4(self.c4(X))))
        X=X.view(-1,2048)
        X=F.relu(self.fc1(F.relu(self.fc0(X))))
        X=self.fc2(X)
        return X
    
    
bs=200
img_transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.RandomChoice([transforms.RandomRotation(5),
                                    transforms.RandomAffine(0,translate=(.1,.1))]),
                                    transforms.ToTensor()])
train_data=DataLoader(cifar10_dataset((X_train,y_train),True,img_transform),batch_size=bs,shuffle=False,num_workers=0)
n=X_test.shape[0]
img_transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])                                 
test_data=DataLoader(cifar10_dataset((X_test,y_test),True,img_transform),batch_size=bs,shuffle=False,num_workers=0)
cnn=CNN()
cnn=cnn.to(device)
opt=optim.Adam(cnn.parameters(),lr=1e-3)
loss=nn.CrossEntropyLoss()
epochs=100
train(cnn,train_data,epochs,opt,loss,test_data,n,10)    


    