# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 08:37:41 2021

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

BATCH_SIZE =200 # Batch Size. Adjust accordingly
NUM_WORKERS = 0 # Number of threads to be used for image loading. Adjust accordingly.

img_transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])

test_data = sys.argv[1] # Path to test csv file
test_dataset = DevanagariDataset(data_csv = test_data, train=False, img_transform=img_transforms)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
cnn=CNN_A().to(device)
cnn.load_state_dict(torch.load(sys.argv[2]))
cnn.eval()

f=open(sys.argv[3],'w')
def test(cnn,test_data,f):
    with torch.no_grad():
        for j,(X,y) in enumerate(test_data):
            y_preds=cnn(X.to(device))
            y_preds=torch.argmax(y_preds,dim=1).squeeze()
            
            for pred in y_preds.cpu().numpy():
                #print(pred)
                f.write(str(pred))
                f.write('\n')   
    #print(i)
test(cnn,test_loader,f)
f.close() 
 
    