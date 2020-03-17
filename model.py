#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import gc
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
import catboost as cbt
import math
from lightgbm.sklearn import LGBMClassifier
from collections import Counter  
import time
from scipy.stats import kurtosis,iqr
from scipy import ptp
from tqdm import tqdm
import datetime
from sklearn.metrics import accuracy_score, roc_auc_score,log_loss,f1_score
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import warnings
import os
import joblib
pd.options.display.max_columns = None
pd.options.display.max_rows = None


# In[2]:


path = os.getcwd()
path


# In[3]:


test = pd.read_csv(path + '/data/test/test49.csv')
print(test.shape)  # (178028, 49)
test.head()


# In[4]:


# 训练模型选取的训练数据列
cols = list(set(test.columns)-set(['manufacturer','model','serial_number','dt']))
print(len(cols))  # 45


# In[5]:


train_t = pd.read_csv(path + '/data/train_t/train_t_time30_49.csv')
print(train_t.shape)  # (32547, 49)
train_t.head()


# In[6]:


train_f = pd.read_csv(path + '/data/train_f/707_29766.csv')
print(train_f.shape)  # (29766, 49)
train_f.head()


# In[7]:


train_tt = train_t[cols]
train_tt['y'] = 1
train_ff = train_f[cols]
train_ff['y'] = 0
print(train_tt.shape)  # (32547, 46)
print(train_ff.shape)  # (29766, 46)
train_tt.head()


# In[8]:


train = pd.concat([train_tt, train_ff])
print(train.shape)  # (62313, 46)


# In[10]:


train.head()


# In[17]:


cols_x = list(set(train_tt.columns)-set(['y']))
print(len(cols_x))
x_train,x_val,y_train,y_val=train_test_split(train[cols_x],train['y'],test_size=0.1,random_state=78)


# In[18]:


llf=lgb.LGBMClassifier(learning_rate=0.001,
                        n_estimators=100,
                        num_leaves=127,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=2019,
                    #     is_unbalenced = 'True',
                        metric=None)
print('************** training **************')
print(x_train.shape,x_val.shape)
llf.fit(x_train,y_train,
        eval_set=[(x_val, y_val)],
        eval_metric='auc',
        early_stopping_rounds=10,
        verbose=10)
weight_lgb=f1_score(y_val,llf.predict(x_val),average='binary')
weight_lgb


# In[ ]:





# In[ ]:




