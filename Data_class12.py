#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from tqdm import tqdm
path = os.getcwd()
path


# # 负样本采样：12万，1万1个月份，
# # 前提去掉raw+normalized:10,240,241,242，
# # 前提筛选条件model==1，
# # 筛选前提条件：1.raw:5,187,188,197,198任一列>0；2.raw:5,187,188,197,198全为0；（1，2各采取12万数据）；
# # 按照smart_1raw列的两个簇划分，每个月每个簇各取5K
# # smart_1raw列划分为[0,100]的簇划分值为47

# In[257]:


# 先求出smart_1raw列两个簇真实的划分值
dt = pd.read_csv(path + '/data/processed_date_data/traint.csv')
print(dt.shape)
dt.tail()


# In[258]:


dt.describe()


# In[259]:


raw1_sp = (max(dt['smart_1raw'])/100) * 47
raw1_sp


# In[260]:


dt0 = dt[dt['smart_1raw'] < raw1_sp]
print(dt0.shape)  # (15136, 52)
dt0.head()


# In[261]:


dt1 = dt[dt['smart_1raw'] >= raw1_sp]
print(dt1.shape)  # (14853, 52)
dt1.head()


# In[9]:


# 抽取第一类12万的负样本：raw:5,187,188,197,198任一列>0；
trainf = []
raw1_sp = 114737911.52000001
del_cols = ['smart_10raw','smart_10_normalized','smart_240raw','smart_240_normalized',
           'smart_241raw','smart_241_normalized','smart_242raw','smart_242_normalized']
path_files = path + '/data/processed_date_data/'
for file in os.listdir(path_files):
    if file[-3:] == 'csv' and file[:2] == '20':
        num0, num1 = 5000, 5000
        tp = pd.read_csv(path_files+file)
        print(file, tp.shape)
        tp = tp[set(tp.columns)-set(del_cols)]  # 删除8个列：raw+normalized:10，240，241，242
        tp = tp[tp['model'] == 1]  # 筛选model==1
        tp = tp[tp['label'] == 0]  # 筛选出负样本
        tpp = tp[(tp['smart_5raw']>0) | 
                (tp['smart_187raw']>0) | 
                (tp['smart_188raw']>0) | 
                (tp['smart_197raw']>0) | 
                (tp['smart_198raw']>0)]  # 筛选raw:5,187,188,197,198任一列>0
        tpp.reset_index(drop=True, inplace=True)  # 一定要重新设定index
        print(tpp.shape)
        tp0 = tpp[tpp['smart_1raw'] < raw1_sp]
        print(tp0.shape)
        tp1 = tpp[tpp['smart_1raw'] >= raw1_sp]
        print(tp1.shape)
        if tp0.shape[0] < num0:
            num0 = tp0.shape[0]
            if tpp.shape[0] < 10000:
                num1 = tpp.shape[0] - num0
            else:
                num1 = 10000 - num0
        elif tp1.shape[0] < num1:
            num1 = tp1.shape[0]
            if tpp.shape[0] < 10000:
                num0 = tpp.shape[0] - num1
            num0 = 10000 - num1
        print(num0, num1)
        ts0 = tp0.sample(n=num0)
        ts1 = tp1.sample(n=num1)
        trainf.append(ts0)
        trainf.append(ts1)
tf = pd.concat(trainf, ignore_index=True)
print(tf.shape)
tf.head()


# In[10]:


tf.to_csv(path + '/data/processed_date_data/trainf1_120k.csv', index=False)


# In[11]:


# 抽取第二类12万的负样本：raw:5,187,188,197,198全为0；
trainf = []
raw1_sp = 114737911.52000001
del_cols = ['smart_10raw','smart_10_normalized','smart_240raw','smart_240_normalized',
           'smart_241raw','smart_241_normalized','smart_242raw','smart_242_normalized']
path_files = path + '/data/processed_date_data/'
for file in os.listdir(path_files):
    if file[-3:] == 'csv' and file[:2] == '20':
        num0, num1 = 5000, 5000
        tp = pd.read_csv(path_files+file)
        print(file, tp.shape)
        tp = tp[set(tp.columns)-set(del_cols)]  # 删除8个列：raw+normalized:10，240，241，242
        tp = tp[tp['model'] == 1]  # 筛选model==1
        tp = tp[tp['label'] == 0]  # 筛选出负样本
        tpp = tp[(tp['smart_5raw']==0) | 
                (tp['smart_187raw']==0) | 
                (tp['smart_188raw']==0) | 
                (tp['smart_197raw']==0) | 
                (tp['smart_198raw']==0)]  # 筛选raw:5,187,188,197,198全为0
        tpp.reset_index(drop=True, inplace=True)  # 一定要重新设定index
        print(tpp.shape)
        tp0 = tpp[tpp['smart_1raw'] < raw1_sp]
        print(tp0.shape)
        tp1 = tpp[tpp['smart_1raw'] >= raw1_sp]
        print(tp1.shape)
        if tp0.shape[0] < num0:
            num0 = tp0.shape[0]
            if tpp.shape[0] < 10000:
                num1 = tpp.shape[0] - num0
            else:
                num1 = 10000 - num0
        elif tp1.shape[0] < num1:
            num1 = tp1.shape[0]
            if tpp.shape[0] < 10000:
                num0 = tpp.shape[0] - num1
            num0 = 10000 - num1
        print(num0, num1)
        ts0 = tp0.sample(n=num0)
        ts1 = tp1.sample(n=num1)
        trainf.append(ts0)
        trainf.append(ts1)
tf = pd.concat(trainf, ignore_index=True)
print(tf.shape)
tf.to_csv(path + '/data/processed_date_data/trainf2_120k.csv', index=False)


# In[15]:


# 抽取第一类正样本：raw:5,187,188,197,198任一列>0；
raw1_sp = 114737911.52000001
del_cols = ['smart_10raw','smart_10_normalized','smart_240raw','smart_240_normalized',
           'smart_241raw','smart_241_normalized','smart_242raw','smart_242_normalized']
traint = pd.read_csv(path + '/data/processed_date_data/traint.csv')
print(traint.shape)  # (29992, 52)
traint = traint[set(traint.columns)-set(del_cols)]  # 删除8个列：raw+normalized:10，240，241，242
traint = traint[traint['model'] == 1]  # 筛选model==1
print(traint.shape)  # (17308, 44)
tt = traint[(traint['smart_5raw']>0) | 
                (traint['smart_187raw']>0) | 
                (traint['smart_188raw']>0) | 
                (traint['smart_197raw']>0) | 
                (traint['smart_198raw']>0)]  # 筛选raw:5,187,188,197,198任一列>0
tt.reset_index(drop=True, inplace=True)  # 一定要重新设定index
print(tt.shape)  # (7505, 44)
tt.to_csv(path + '/data/processed_date_data/traint1.csv', index=False)


# In[16]:


# 抽取第二类正样本：raw:5,187,188,197,198全为0；
raw1_sp = 114737911.52000001
del_cols = ['smart_10raw','smart_10_normalized','smart_240raw','smart_240_normalized',
           'smart_241raw','smart_241_normalized','smart_242raw','smart_242_normalized']
traint = pd.read_csv(path + '/data/processed_date_data/traint.csv')
print(traint.shape)  # (29992, 52)
traint = traint[set(traint.columns)-set(del_cols)]  # 删除8个列：raw+normalized:10，240，241，242
traint = traint[traint['model'] == 1]  # 筛选model==1
print(traint.shape)  # (17308, 44)
tt = traint[(traint['smart_5raw']==0) | 
                (traint['smart_187raw']==0) | 
                (traint['smart_188raw']==0) | 
                (traint['smart_197raw']==0) | 
                (traint['smart_198raw']==0)]  # 筛选raw:5,187,188,197,198全为0
tt.reset_index(drop=True, inplace=True)  # 一定要重新设定index
print(tt.shape)  # (16919, 44)
tt.to_csv(path + '/data/processed_date_data/traint2.csv', index=False)


# In[17]:


# 抽取第一类测试集：raw:5,187,188,197,198任一列>0；
raw1_sp = 114737911.52000001
del_cols = ['smart_10raw','smart_10_normalized','smart_240raw','smart_240_normalized',
           'smart_241raw','smart_241_normalized','smart_242raw','smart_242_normalized']
data = pd.read_csv(path + '/data/processed_date_data/test.csv')
print(data.shape)  # (178028, 49)
data = data[set(data.columns)-set(del_cols)]  # 删除8个列：raw+normalized:10，240，241，242
dt = data[(data['smart_5raw']>0) | 
                (data['smart_187raw']>0) | 
                (data['smart_188raw']>0) | 
                (data['smart_197raw']>0) | 
                (data['smart_198raw']>0)]  # 筛选raw:5,187,188,197,198任一列>0
dt.reset_index(drop=True, inplace=True)  # 一定要重新设定index
print(dt.shape)  # (33111, 41)
dt.to_csv(path + '/data/processed_date_data/test1.csv', index=False)


# In[18]:


# 抽取第二类测试集：raw:5,187,188,197,198全为0；
raw1_sp = 114737911.52000001
del_cols = ['smart_10raw','smart_10_normalized','smart_240raw','smart_240_normalized',
           'smart_241raw','smart_241_normalized','smart_242raw','smart_242_normalized']
data = pd.read_csv(path + '/data/processed_date_data/test.csv')
print(data.shape)  # (178028, 49)
data = data[set(data.columns)-set(del_cols)]  # 删除8个列：raw+normalized:10，240，241，242
dt = data[(data['smart_5raw']==0) | 
                (data['smart_187raw']==0) | 
                (data['smart_188raw']==0) | 
                (data['smart_197raw']==0) | 
                (data['smart_198raw']==0)]  # 筛选raw:5,187,188,197,198全为0
dt.reset_index(drop=True, inplace=True)  # 一定要重新设定index
print(dt.shape)  # (177946, 41)
dt.to_csv(path + '/data/processed_date_data/test2.csv', index=False)


# In[20]:


# 用disk_first.csv每个硬盘观察的第一天与正负样本每行匹配获得每个硬盘当时的观察天数：dt-disk_first.dt
data = pd.read_csv(path + '/data/processed_date_data/disk_first.csv')
print(data.shape)  # (174917, 41)
data = data[data['model']==1]
print(data.shape)  # (102630, 41)
dt = data[['serial_number','dt']]
dt['dt_first'] = dt['dt']
del dt['dt']
dt.head()


# In[22]:


dt.to_csv(path +'/data/processed_date_data/disk1_dt_first.csv', index=False)


# In[1]:


# 第一类正样本构建硬盘观察天数
tt = pd.read_csv(path + '/data/processed_date_data/traint1.csv')
print(tt.shape)  # (7505, 44)
tt = tt.merge(dt, how='left',on='serial_number')
tt['days'] = (pd.to_datetime(tt['dt'])-pd.to_datetime(tt['dt_first'])).dt.days
tt.to_csv(path + '/data/processed_date_data/traintd1.csv', index=False)
print(tt.shape)  # (7505, 46)
tt.head()


# In[31]:


# 第二类正样本构建硬盘观察天数
tt = pd.read_csv(path + '/data/processed_date_data/traint2.csv')
print(tt.shape)  # (16919, 44)
tt = tt.merge(dt, how='left',on='serial_number')
tt['days'] = (pd.to_datetime(tt['dt'])-pd.to_datetime(tt['dt_first'])).dt.days
tt.to_csv(path + '/data/processed_date_data/traintd2.csv', index=False)
print(tt.shape)  # (16919, 46)
tt.head()


# In[30]:


# 第一类负样本构建硬盘观察天数
tt = pd.read_csv(path + '/data/processed_date_data/trainf1_120k.csv')
print(tt.shape)  # (120000, 44)
tt = tt.merge(dt, how='left',on='serial_number')
tt['days'] = (pd.to_datetime(tt['dt'])-pd.to_datetime(tt['dt_first'])).dt.days
tt.to_csv(path + '/data/processed_date_data/trainfd1.csv', index=False)
print(tt.shape)  # (120000, 46)
tt.head()


# In[29]:


# 第二类负样本构建硬盘观察天数
tt = pd.read_csv(path + '/data/processed_date_data/trainf2_120k.csv')
print(tt.shape)  # (120000, 44)
tt = tt.merge(dt, how='left',on='serial_number')
tt['days'] = (pd.to_datetime(tt['dt'])-pd.to_datetime(tt['dt_first'])).dt.days
tt.to_csv(path + '/data/processed_date_data/trainfd2.csv', index=False)
print(tt.shape)  # (120000, 46)
tt.head()


# In[33]:


# 第一类测试集构建硬盘观察天数
tt = pd.read_csv(path + '/data/processed_date_data/test1.csv')
print(tt.shape)  # (33111, 41)
tt = tt.merge(dt, how='left',on='serial_number')
tt['days'] = (pd.to_datetime(tt['dt'])-pd.to_datetime(tt['dt_first'])).dt.days
tt.to_csv(path + '/data/processed_date_data/testd1.csv', index=False)
print(tt.shape)  # (33111, 43)
tt.head()


# In[34]:


# 第二类测试集构建硬盘观察天数
tt = pd.read_csv(path + '/data/processed_date_data/test2.csv')
print(tt.shape)  # (177946, 41)
tt = tt.merge(dt, how='left',on='serial_number')
tt['days'] = (pd.to_datetime(tt['dt'])-pd.to_datetime(tt['dt_first'])).dt.days
tt.to_csv(path + '/data/processed_date_data/testd2.csv', index=False)
print(tt.shape)  # (177946, 43)
tt.head()


# In[ ]:




