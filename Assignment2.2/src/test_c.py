# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 11:13:55 2021

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

#X_test=pd.read_csv(sys.argv[1],header=None)
#X_test,y_test=np.array(X_test.iloc[:,1:],dtype=np.uint8),np.array(X_test.iloc[:,0],dtype=int)
#X_test=X_test.reshape((-1,3,32,32)).transpose(0,2,3,1)    

    
X_test=pd.read_csv(sys.argv[1],header=None)
X_test=np.array(X_test.iloc[:,:],dtype=np.uint8)
X_test=X_test.reshape((-1,3,32,32)).transpose(0,2,3,1)
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

bs=146
img_transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])                                 
#test_data=DataLoader(cifar10_dataset((X_test,y_test),True,img_transform),batch_size=bs,shuffle=False,num_workers=20)
test_data=DataLoader(cifar10_dataset(X_test,False,img_transform),batch_size=bs,shuffle=False,num_workers=0)
cnn=CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cnn=cnn.to(device)
cnn.load_state_dict(torch.load(sys.argv[2]))
cnn.eval()

n=X_test.shape[0]
#print(X_test.shape)
f=open(sys.argv[3],'w')

def test(cnn,test_data,f,n):
    with torch.no_grad():
        a=0
        for j,(X,y) in enumerate(test_data):
            #print(X.shape)
            y_preds=cnn(X.to(device))
            y_preds=torch.argmax(y_preds,dim=1).squeeze()
            #a+=(y.cuda()==y_preds).sum().item()
            for pred in y_preds.cpu().numpy():
                #print(str(pred))
                f.write(str(pred)+"\n")   
    #print(a/n)
test(cnn,test_data,f,n)  
f.close()