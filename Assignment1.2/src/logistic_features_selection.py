# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 20:18:57 2021

@author: Asus
"""

import numpy as np
import pandas as pd
#from numpy.linalg import norm      
#from boruta import BorutaPy
#from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest,f_classif, mutual_info_classif,chi2,SelectFdr,SelectFpr,SelectFwe,SelectPercentile
#from category_encoders.target_encoder import TargetEncoder 
#from category_encoders.cat_boost import CatBoostEncoder 
#from category_encoders.basen import BaseNEncoder 
#from category_encoders.backward_difference import BackwardDifferenceEncoder
#from category_encoders.glmm import GLMMEncoder
#from category_encoders.helmert import HelmertEncoder
#from category_encoders.wrapper import NestedCVWrapper, PolynomialWrapper
#from category_encoders.count import CountEncoder
#from category_encoders.james_stein import JamesSteinEncoder
#from category_encoders.polynomial import PolynomialEncoder
#from category_encoders.sum_coding import SumEncoder
#from category_encoders.woe import WOEEncoder
#from sklearn.preprocessing import PolynomialFeatures
#from scipy.special import softmax as sm
import sys

def agg_cat(df,cat_col,num_col):
    "min"
    a = df.groupby(cat_col)[num_col].agg(["std","max","mean"]).reset_index()
    a.columns = [cat_col] + ["grpby_"+ str(cat_col)+ "_" + str(a.columns[i]) +  "_" + str(num_col) for i in range(1,len(a.columns))]
    return df.merge(a,on = cat_col, how="left")
# ,pd.Series.nunique
def agg_cat1(df,cat_col,num_col):
    a = df.groupby(cat_col)[num_col].agg([pd.Series.mode]).reset_index()
    a.columns = [cat_col] + ["grpby_"+ str(cat_col)+ "_" + str(a.columns[i]) +  "_" + str(num_col) for i in range(1,len(a.columns))]
    return df.merge(a,on = cat_col, how="left")


train_path = sys.argv[1]
test_path = sys.argv[2]

train_df = pd.read_csv(train_path, index_col = 0)    
test_df = pd.read_csv(test_path, index_col = 0)
train_df.insert(0, 'Intercept', 1)
test_df.insert(0, 'Intercept', 1)

#y_train = np.array(train['Length of Stay'])
y_train=pd.get_dummies(train_df['Length of Stay']).to_numpy()


train_df=train_df.drop(columns = ['Length of Stay'])

# for c in train.columns:
#     train[c] = (train[c]-train[c].mean())/train[c].std() 


X_train=train_df.to_numpy()
X_test=test_df.to_numpy()


fs = SelectKBest(score_func=f_classif, k='all')
fs.fit(X_train,np.argmax(y_train,axis=1)+1)
X_train_fs = fs.transform(X_train)
# X_test_fs = fs.transform(X_test)
selected_features_1=list(np.argsort(fs.scores_))[::-1]

print("Most Predictive Features")
print(dict(zip(train_df.columns[selected_features_1],fs.scores_[selected_features_1])))

train_path = sys.argv[1]
test_path = sys.argv[2]

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


#Ensuring consistency of One-Hot Encoding
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
X_train.shape

X_train=temp.to_numpy()
X_test=test_temp.to_numpy()


X_train.shape,X_test.shape

# ###FEATURE CREATION
# # temp=train_df.iloc[:,np.array(selected_features_1)[:]].to_numpy()
# #temp=X_train
# # polyfeaturecreator = PolynomialFeatures(2)
# # temp=polyfeaturecreator.fit_transform(temp)
# # X_train=temp
# X_train=np.concatenate((X_train,
#                agg_cat(train_df,'APR Medical Surgical Description','Total Costs').iloc[:,-1:].to_numpy(),
#                agg_cat(train_df,'Emergency Department Indicator','Total Costs').iloc[:,-1:].to_numpy(),
#                agg_cat(train_df,'APR Severity of Illness Code','Total Costs').iloc[:,-1:].to_numpy(),
#                agg_cat(train_df,'APR Risk of Mortality','Total Costs').iloc[:,-1:].to_numpy(),
#                agg_cat(train_df,'Age Group','Total Costs').iloc[:,-1:].to_numpy(),
#                agg_cat(train_df,'APR DRG Code','Total Costs').iloc[:,-1:].to_numpy(),
#                agg_cat(train_df,'APR MDC Code','Total Costs').iloc[:,-1:].to_numpy(),
#                agg_cat(train_df,'CCS Procedure Code','Total Costs').iloc[:,-1:].to_numpy(),
#                agg_cat(train_df,'CCS Diagnosis Code','Total Costs').iloc[:,-1:].to_numpy(),
#                agg_cat(train_df,'Birth Weight','Total Costs').iloc[:,-1:].to_numpy(),
#                agg_cat(train_df,'Patient Disposition','Total Costs').iloc[:,-1:].to_numpy(),
#                agg_cat(train_df,'Gender','Total Costs').iloc[:,-1:].to_numpy(),
#                agg_cat(train_df,'Payment Typology 1','Total Costs').iloc[:,-1:].to_numpy(),
#                agg_cat(train_df,'Type of Admission','Total Costs').iloc[:,-1:].to_numpy(),
#                agg_cat(train_df,'Payment Typology 3','Total Costs').iloc[:,-1:].to_numpy(),
#                agg_cat(train_df,'Ethnicity','Total Costs').iloc[:,-1:].to_numpy()            
#                     ),axis=1)

                

# X_train=np.concatenate((X_train,np.exp(train_df['Total Costs'].to_numpy()).reshape(-1,1),
#                         np.square(train_df['Total Costs'].to_numpy()).reshape(-1,1)
#                        ),axis=1)
# X_train.shape
selector = SelectKBest(score_func=f_classif, k='all')

selector.fit(X_train,np.argmax(y_train,axis=1)+1)

X_train = selector.transform(X_train)
# X_test = selector.transform(X_test)
selected_features=list(np.argsort(selector.scores_))[::-1][:500]
X_train=X_train[:,selected_features]
X_test=X_test[:,selected_features]
X_train.shape

print("Indices of Selected Features (<=500)")
print(selected_features)

def softmax(a):
    m=np.max(a,axis=1).reshape(-1,1)
    #gamma=1e-15
    a=np.exp(a-m)
    #return a/(a.sum(axis=a.ndim-1).reshape(-1,1))
    return np.nan_to_num(a/(a.sum(axis=a.ndim-1).reshape(-1,1)),False)
def CE_loss(y_true,y_preds):
    gamma=1e-15
    return -np.sum(np.multiply(y_true,np.log(np.clip(y_preds,gamma,1-gamma))))/y_true.shape[0]

def acc(y_true,y_preds):
    y_preds=np.argmax(y_preds,axis=1).squeeze()
    y_true=np.argmax(y_true,axis=1).squeeze()
    return np.count_nonzero(y_true-y_preds==0)/len(y_true)


def grad_desc(bs,num_iter,lr,X_train,y_train):
    W=np.zeros((X_train.shape[1],8))
    for i in range(num_iter):
        for j in range(X_train.shape[0]//bs):
            #print(X_train[j*bs:(j+1)*bs,:].shape,y_train[j*bs:(j+1)*bs,:].shape)
            #print(softmax(np.matmul(X_train,W)))
            #print(i)
            #print(np.matmul(X_train,W))
            x,y=X_train[j*bs:(j+1)*bs,:],y_train[j*bs:(j+1)*bs,:]
            W-=lr/(bs)*(np.matmul(x.T,softmax(np.matmul(x,W))-y))
            
            #print((np.matmul(X_train.T,y_train-softmax(np.matmul(X_train,W)))).shape)
        y_pred=softmax(np.matmul(X_train,W))
        #print(np.matmul(X_train,W)[0])
#         print(softmax(np.matmul(X_train,W))[0])
        print(i,CE_loss(y_train,y_pred),acc(y_train,y_pred))
        
    return W

def adaptive_grad_desc(bs,num_iter,lr,X_train,y_train):
    W=np.zeros((X_train.shape[1],8))
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
            print(W)
    return W

def backtrack_grad_desc(bs,num_iter,lr,alpha,beta,X_train,y_train):
        
        W=np.zeros((X_train.shape[1],8))
        lr_=lr
        for i in range(num_iter):
        #    if(i%100==0):
         #       print(i)
            yh=softmax(np.matmul(X_train,W))
            dw=np.matmul(X_train.T,yh-y_train)/(y_train.shape[0])
            
            while(CE_loss(y_train,softmax(np.matmul(X_train,W-lr*dw)))-CE_loss(y_train,yh)>
                 -alpha*lr*(np.square(np.linalg.norm(dw)))):
                lr*=beta
                
            print("Current LR:" +str(lr))
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
            y_pred=softmax(np.matmul(X_train,W))
            print(i,CE_loss(y_train,y_pred),acc(y_train,y_pred))    
            #print((np.matmul(X_train.T,y_train-softmax(np.matmul(X_train,W)))).shape)
        return W
    
bs=300
#W=grad_desc(bs,500,.05,X_train,y_train)
W=backtrack_grad_desc(bs,750,1.55,0.5,.91,X_train,y_train)

y_pred=softmax(np.matmul(X_test,W))
y_pred=np.squeeze(np.argmax(y_pred,axis=1)+1)



f=open(sys.argv[3],'w')
for y in y_pred:
    f.write(str(y)+'\n')
f.close()
        
f=open(sys.argv[4],'w')
for w in W.flatten().squeeze():
    f.write(str(w)+'\n')
f.close()
    

