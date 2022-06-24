# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 08:20:01 2021

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
import sys

# DataLoader Class
# if BATCH_SIZE = N, dataloader returns images tensor of size [N, C, H, W] and labels [N]
class DevanagariDataset(Dataset):
    
    def __init__(self, data_csv, train = True , img_transform = None):
        """
        Dataset init function
        
        INPUT:
        data_csv: Path to csv file containing [data, labels]
        train: 
            True: if the csv file has [data, labels] (Train data and Public Test Data) 
            False: if the csv file has only [data] and labels are not present.
        img_transform: List of preprocessing operations need to performed on image. 
        """
        self.data_csv = data_csv
        self.img_transform = img_transform
        self.is_train = train
        
        data = pd.read_csv(data_csv, header=None)
        if self.is_train:
            images = data.iloc[:,:-1].to_numpy()
            labels = data.iloc[:,-1].astype(int)
        else:
            images = data.iloc[:,:].to_numpy()
            labels = None
        
        self.images = images
        self.labels = labels
        #print("Total Images: {}, Data Shape = {}".format(len(self.images), images.shape))
        
    def __len__(self):
        """Returns total number of samples in the dataset"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Loads image of the given index and performs preprocessing.
        
        INPUT: 
        idx: index of the image to be loaded.
        
        OUTPUT:
        sample: dictionary with keys images (Tensor of shape [1,C,H,W]) and labels (Tensor of labels [1]).
        """
        image = self.images[idx]
        image = np.array(image).astype(np.uint8).reshape(32, 32, 1)
        
        if self.is_train:
            label = self.labels[idx]
        else:
            label = -1
        
        image = self.img_transform(image)
#         print(image.shape, label, type(image))
        #sample = {"images": image, "labels": label}
        return image,label    

# Data Loader Usage

args=sys.argv

BATCH_SIZE = 200 # Batch Size. Adjust accordingly
NUM_WORKERS = 0 # Number of threads to be used for image loading. Adjust accordingly.

img_transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])

# Train DataLoader
train_data = args[1] # Path to train csv file
train_dataset = DevanagariDataset(data_csv = train_data, train=True, img_transform=img_transforms)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)

# Test DataLoader
test_data = args[2] # Path to test csv file
test_dataset = DevanagariDataset(data_csv = test_data, train=True, img_transform=img_transforms)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)
n=test_dataset.__len__()

class CNN_A(nn.Module):
    def __init__(self):
        super(CNN_A,self).__init__()
        self.c1=nn.Conv2d(1,32,3)
        self.bn1=nn.BatchNorm2d(32)
        self.c2=nn.Conv2d(32,64,3)
        self.bn2=nn.BatchNorm2d(64)
        self.c3=nn.Conv2d(64,256,3)
        self.bn3=nn.BatchNorm2d(256)
        self.c4=nn.Conv2d(256,512,3)
        self.fc1=nn.Linear(512,256)
        self.fc2=nn.Linear(256,46)
        self.dropout=nn.Dropout(.2)
        self.p1=nn.MaxPool2d(2,1)
        self.p2=nn.MaxPool2d(2,2)
    def forward(self,X):
        X=self.p2(F.relu(self.bn1(self.c1(X))))
        X=self.p2(F.relu(self.bn2(self.c2(X)))) 
        X=self.p1(F.relu(self.bn3(self.c3(X))))        
        X=F.relu(self.c4(X))
        X=X.view(-1,512)
        X=self.fc2(self.dropout(F.relu(self.fc1(X))))
        return X
def acc(y_true,y_preds):
    y_preds=torch.argmax(y_preds,dim=1).squeeze()
    return (y_true==y_preds).sum().item()/len(y_true)



#train_data=DataLoader(dev_dataset(X_train,y_train),batch_size=bs,shuffle=False)
# X_train,y_train=torch.from_numpy(X_train).cuda(),torch.LongTensor(y_train).cuda()
#X_test,y_test=torch.from_numpy(X_test).cuda(),torch.LongTensor(y_test).cuda()
torch.manual_seed(51)
cnn=CNN_A()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn=cnn.to(device)
opt=optim.Adam(cnn.parameters(),lr=1e-4)
loss=nn.CrossEntropyLoss()
def train(cnn,train_data,epochs,opt,loss,test_data,n):
    losses=[]
    accuracies=[]
    for i in range(epochs):
        l=0
        for j,(X,y) in enumerate(train_data):
            #print(X.shape,y.shape,type(X),type(y))
            yh=cnn(X.to(device))
            train_loss=loss(yh,y.type(torch.LongTensor).to(device))
            
            
            train_loss.backward()
            opt.step()
            opt.zero_grad()
            with torch.no_grad():
                l+=train_loss.item()
        with torch.no_grad():
            l/=len(train_data)
            losses.append(l)
        
            c=0
            for j,(X,y) in enumerate(test_data):
            
                
                y_preds=cnn(X.to(device))
                y_preds=torch.argmax(y_preds,dim=1).squeeze()
                c+=(y.to(device)==y_preds).sum().item()
        
            c/=n    
        accuracies.append(c)
        torch.save(cnn.state_dict(),args[3]) 
        #print("Epoch: "+str(i+1)+" Train loss: "+str(losses[-1])+" test accuracy: " +str(accuracies[-1]))        
    return losses,accuracies      
            

epochs=8
losses,accuracies=train(cnn,train_loader,epochs,opt,loss,test_loader,n)

f=open(args[4],'w')
for l in losses:
    f.write(str(l))
    f.write('\n')
f.close()    
f=open(args[5],'w')
for l in accuracies:
    f.write(str(l))
    f.write('\n')
f.close()
torch.save(cnn.state_dict(),args[3]) 

