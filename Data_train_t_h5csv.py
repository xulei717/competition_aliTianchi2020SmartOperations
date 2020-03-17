#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
path = os.getcwd()


# In[2]:


ft = pd.read_csv(path + '/data/disk_sample_fault_tag.csv')
ft.head(1)


# In[3]:


ftd = {}
for i in range(ft.shape[0]):
    k = ''.join([ft.loc[i,'manufacturer'],str(ft.loc[i,'model']),ft.loc[i,'serial_number']])
    ftd[k] = i
print(len(ftd), ftd['A1disk_100102'])


# In[5]:


dt = pd.read_csv(path + '/data/disk_sample_smart_log_2017/disk_sample_smart_log_201707.csv')
print(dt.shape)
dt = dt.iloc[:, 1:]
print(dt.shape)  # (1610007, 514)
dt.head()
cols = [x for x in dt.columns if x != 'dt']
print(len(cols), 'dt' in cols)


# In[26]:


dt_dup = dt.drop_duplicates(subset=cols)
print(dt_dup.shape)  # (1610007, 514)
dt_dup.head()


# In[28]:


train_t201707 = []
for i in range(dt.shape[0]):
    k = ''.join([dt.loc[i,'manufacturer'],str(dt.loc[i,'model']),dt.loc[i,'serial_number']])
    if k in ftd:
        j = ftd[k]
        tmp = dt.loc[i:i, :]
        tmp['fault_time'] = ft.loc[j, 'fault_time']
        tmp['tag'] = ft.loc[j, 'tag']
        # print(tmp)
        train_t201707.append(tmp)
train_ts707 = pd.concat(train_t201707)
print(len(train_ts707))  # 11791
train_ts707.head()


# In[31]:


train_ts707.to_hdf(path + '/data/train_t/train_ts707.h5', 'df', mode='w')


# In[ ]:





# In[5]:


# disk_sample_smart_log_201708.csv
dt = pd.read_csv(path + '/data/disk_sample_smart_log_2017/disk_sample_smart_log_201708.csv')
print(dt.shape)
dt = dt.iloc[:, 1:]
print(dt.shape)  # (3140857, 514)
dt.head()
cols = [x for x in dt.columns if x != 'dt']
print(len(cols), 'dt' in cols)
dt_dup = dt.drop_duplicates(subset=cols)
print(dt_dup.shape)  # (3140857, 514)
dt_dup.head()


# In[9]:


# train_t/train_ts708.h5
train_t = []
for i in range(dt.shape[0]):
    k = ''.join([dt.loc[i,'manufacturer'],str(dt.loc[i,'model']),dt.loc[i,'serial_number']])
    if k in ftd:
        j = ftd[k]
        tmp = dt.loc[i:i, :]
        tmp['fault_time'] = ft.loc[j, 'fault_time']
        tmp['tag'] = ft.loc[j, 'tag']
        # print(tmp)
        train_t.append(tmp)
train_ts = pd.concat(train_t)
train_ts.to_hdf(path + '/data/train_t/train_ts708.h5', 'df', mode='w')
print(len(train_ts))  # 22038
train_ts.head()


# In[10]:


# disk_sample_smart_log_201709.csv
dt = pd.read_csv(path + '/data/disk_sample_smart_log_2017/disk_sample_smart_log_201709.csv')
print(dt.shape)
dt = dt.iloc[:, 1:]
print(dt.shape)  # (3146965, 514)
dt.head()
cols = [x for x in dt.columns if x != 'dt']
print(len(cols), 'dt' in cols)
dt_dup = dt.drop_duplicates(subset=cols)
print(dt_dup.shape)  # (3146965, 514)
dt_dup.head()


# In[11]:


# train_t/train_ts709.h5
train_t = []
for i in range(dt.shape[0]):
    k = ''.join([dt.loc[i,'manufacturer'],str(dt.loc[i,'model']),dt.loc[i,'serial_number']])
    if k in ftd:
        j = ftd[k]
        tmp = dt.loc[i:i, :]
        tmp['fault_time'] = ft.loc[j, 'fault_time']
        tmp['tag'] = ft.loc[j, 'tag']
        # print(tmp)
        train_t.append(tmp)
train_ts = pd.concat(train_t)
train_ts.to_hdf(path + '/data/train_t/train_ts709.h5', 'df', mode='w')
print(len(train_ts))  # 20266
train_ts.head()


# In[7]:


# disk_sample_smart_log_201710.csv
dt = pd.read_csv(path + '/data/disk_sample_smart_log_2017/disk_sample_smart_log_201710.csv')
print(dt.shape)
dt = dt.iloc[:, 1:]
print(dt.shape)  # (3536616, 514)
dt.head()
cols = [x for x in dt.columns if x != 'dt']
print(len(cols), 'dt' in cols)
dt_dup = dt.drop_duplicates(subset=cols)
print(dt_dup.shape)  # (3536616, 514)
dt_dup.head()


# In[8]:


# train_t/train_ts710.h5
train_t = []
for i in range(dt.shape[0]):
    k = ''.join([dt.loc[i,'manufacturer'],str(dt.loc[i,'model']),dt.loc[i,'serial_number']])
    if k in ftd:
        j = ftd[k]
        tmp = dt.loc[i:i, :]
        tmp['fault_time'] = ft.loc[j, 'fault_time']
        tmp['tag'] = ft.loc[j, 'tag']
        # print(tmp)
        train_t.append(tmp)
train_ts = pd.concat(train_t)
train_ts.to_hdf(path + '/data/train_t/train_ts710.h5', 'df', mode='w')
print(len(train_ts))  # 21059
train_ts.head()


# In[6]:


# disk_sample_smart_log_201711.csv
dt = pd.read_csv(path + '/data/disk_sample_smart_log_2017/disk_sample_smart_log_201711.csv')
print(dt.shape)
dt = dt.iloc[:, 1:]
print(dt.shape)  # (3639220, 514)
dt.head()
cols = [x for x in dt.columns if x != 'dt']
print(len(cols), 'dt' in cols)
dt_dup = dt.drop_duplicates(subset=cols)
print(dt_dup.shape)  # (3639220, 514)
dt_dup.head()


# In[7]:


# train_t/train_ts711.h5
train_t = []
for i in range(dt.shape[0]):
    k = ''.join([dt.loc[i,'manufacturer'],str(dt.loc[i,'model']),dt.loc[i,'serial_number']])
    if k in ftd:
        j = ftd[k]
        tmp = dt.loc[i:i, :]
        tmp['fault_time'] = ft.loc[j, 'fault_time']
        tmp['tag'] = ft.loc[j, 'tag']
        # print(tmp)
        train_t.append(tmp)
train_ts = pd.concat(train_t)
train_ts.to_hdf(path + '/data/train_t/train_ts711.h5', 'df', mode='w')
print(len(train_ts))  # 19806
train_ts.head()


# In[4]:


# disk_sample_smart_log_201712.csv
dt = pd.read_csv(path + '/data/disk_sample_smart_log_2017/disk_sample_smart_log_201712.csv')
print(dt.shape)
dt = dt.iloc[:, 1:]
print(dt.shape)  # (4105046, 514)
dt.head()
cols = [x for x in dt.columns if x != 'dt']
print(len(cols), 'dt' in cols)
dt_dup = dt.drop_duplicates(subset=cols)
print(dt_dup.shape)  # (4105046, 514)
dt_dup.head()


# In[5]:


# train_t/train_ts712.h5
train_t = []
for i in range(dt.shape[0]):
    k = ''.join([dt.loc[i,'manufacturer'],str(dt.loc[i,'model']),dt.loc[i,'serial_number']])
    if k in ftd:
        j = ftd[k]
        tmp = dt.loc[i:i, :]
        tmp['fault_time'] = ft.loc[j, 'fault_time']
        tmp['tag'] = ft.loc[j, 'tag']
        # print(tmp)
        train_t.append(tmp)
train_ts = pd.concat(train_t)
train_ts.to_hdf(path + '/data/train_t/train_ts712.h5', 'df', mode='w')
print(len(train_ts))  # 21600
train_ts.head()


# In[4]:


# disk_sample_smart_log_201801.csv
dt = pd.read_csv(path + '/data/disk_sample_smart_log_2018_Q1/disk_sample_smart_log_201801.csv')
print(dt.shape)
dt = dt.iloc[:, 1:]
print(dt.shape)  # (4428248, 514)
dt.head()
cols = [x for x in dt.columns if x != 'dt']
print(len(cols), 'dt' in cols)
dt_dup = dt.drop_duplicates(subset=cols)
print(dt_dup.shape)  # (4428248, 514)
dt_dup.head()


# In[5]:


# train_t/train_ts801.h5
train_t = []
for i in range(dt.shape[0]):
    k = ''.join([dt.loc[i,'manufacturer'],str(dt.loc[i,'model']),dt.loc[i,'serial_number']])
    if k in ftd:
        j = ftd[k]
        tmp = dt.loc[i:i, :]
        tmp['fault_time'] = ft.loc[j, 'fault_time']
        tmp['tag'] = ft.loc[j, 'tag']
        # print(tmp)
        train_t.append(tmp)
train_ts = pd.concat(train_t)
train_ts.to_hdf(path + '/data/train_t/train_ts801.h5', 'df', mode='w')
print(len(train_ts))  # 21729
train_ts.head()


# In[5]:


# disk_sample_smart_log_201802.csv
dt = pd.read_csv(path + '/data/disk_sample_smart_log_2018_Q1/disk_sample_smart_log_201802.csv')
print(dt.shape)
dt = dt.iloc[:, 1:]
print(dt.shape)  # (4083439, 514)
dt.head()


# In[6]:


# train_t/train_ts802.h5
train_t = []
for i in range(dt.shape[0]):
    k = ''.join([dt.loc[i,'manufacturer'],str(dt.loc[i,'model']),dt.loc[i,'serial_number']])
    if k in ftd:
        j = ftd[k]
        tmp = dt.loc[i:i, :]
        tmp['fault_time'] = ft.loc[j, 'fault_time']
        tmp['tag'] = ft.loc[j, 'tag']
        # print(tmp)
        train_t.append(tmp)
train_ts = pd.concat(train_t)
train_ts.to_hdf(path + '/data/train_t/train_ts802.h5', 'df', mode='w')
print(len(train_ts))  # 16778
train_ts.head()


# In[7]:


# disk_sample_smart_log_201803.csv
dt = pd.read_csv(path + '/data/disk_sample_smart_log_2018_Q1/disk_sample_smart_log_201803.csv')
print(dt.shape)
dt = dt.iloc[:, 1:]
print(dt.shape)  # (4566413, 514)
dt.head()
# train_t/train_ts803.h5
train_t = []
for i in range(dt.shape[0]):
    k = ''.join([dt.loc[i,'manufacturer'],str(dt.loc[i,'model']),dt.loc[i,'serial_number']])
    if k in ftd:
        j = ftd[k]
        tmp = dt.loc[i:i, :]
        tmp['fault_time'] = ft.loc[j, 'fault_time']
        tmp['tag'] = ft.loc[j, 'tag']
        # print(tmp)
        train_t.append(tmp)
train_ts = pd.concat(train_t)
train_ts.to_hdf(path + '/data/train_t/train_ts803.h5', 'df', mode='w')
print(len(train_ts))  # 15626
train_ts.head()


# In[4]:


# disk_sample_smart_log_201804.csv
dt = pd.read_csv(path + '/data/disk_sample_smart_log_2018_Q2/disk_sample_smart_log_201804.csv')
print(dt.shape)
dt = dt.iloc[:, 1:]
print(dt.shape)  # (4463367, 514)
dt.head()
# train_t/train_ts804.h5
train_t = []
for i in range(dt.shape[0]):
    k = ''.join([dt.loc[i,'manufacturer'],str(dt.loc[i,'model']),dt.loc[i,'serial_number']])
    if k in ftd:
        j = ftd[k]
        tmp = dt.loc[i:i, :]
        tmp['fault_time'] = ft.loc[j, 'fault_time']
        tmp['tag'] = ft.loc[j, 'tag']
        # print(tmp)
        train_t.append(tmp)
train_ts = pd.concat(train_t)
train_ts.to_hdf(path + '/data/train_t/train_ts804.h5', 'df', mode='w')
print(len(train_ts))  # 12326
train_ts.head()

# disk_sample_smart_log_201805.csv
dt = pd.read_csv(path + '/data/disk_sample_smart_log_2018_Q2/disk_sample_smart_log_201805.csv')
print(dt.shape)
dt = dt.iloc[:, 1:]
print(dt.shape)  # (4647989, 514)
dt.head()
# train_t/train_ts805.h5
train_t = []
for i in range(dt.shape[0]):
    k = ''.join([dt.loc[i,'manufacturer'],str(dt.loc[i,'model']),dt.loc[i,'serial_number']])
    if k in ftd:
        j = ftd[k]
        tmp = dt.loc[i:i, :]
        tmp['fault_time'] = ft.loc[j, 'fault_time']
        tmp['tag'] = ft.loc[j, 'tag']
        # print(tmp)
        train_t.append(tmp)
train_ts = pd.concat(train_t)
train_ts.to_hdf(path + '/data/train_t/train_ts805.h5', 'df', mode='w')
print(len(train_ts))  # 9768
train_ts.head()

# disk_sample_smart_log_201806.csv
dt = pd.read_csv(path + '/data/disk_sample_smart_log_2018_Q2/disk_sample_smart_log_201806.csv')
print(dt.shape)
dt = dt.iloc[:, 1:]
print(dt.shape)  # (4647434, 514)
dt.head()
# train_t/train_ts806.h5
train_t = []
for i in range(dt.shape[0]):
    k = ''.join([dt.loc[i,'manufacturer'],str(dt.loc[i,'model']),dt.loc[i,'serial_number']])
    if k in ftd:
        j = ftd[k]
        tmp = dt.loc[i:i, :]
        tmp['fault_time'] = ft.loc[j, 'fault_time']
        tmp['tag'] = ft.loc[j, 'tag']
        # print(tmp)
        train_t.append(tmp)
train_ts = pd.concat(train_t)
train_ts.to_hdf(path + '/data/train_t/train_ts806.h5', 'df', mode='w')
print(len(train_ts))  # 6522
train_ts.head()

# disk_sample_smart_log_201807.csv
dt = pd.read_csv(path + '/data/disk_sample_smart_log_2018_Q2/disk_sample_smart_log_201807.csv')
print(dt.shape)
dt = dt.iloc[:, 1:]
print(dt.shape)  # (4855732, 514)
dt.head()
# train_t/train_ts807.h5
train_t = []
for i in range(dt.shape[0]):
    k = ''.join([dt.loc[i,'manufacturer'],str(dt.loc[i,'model']),dt.loc[i,'serial_number']])
    if k in ftd:
        j = ftd[k]
        tmp = dt.loc[i:i, :]
        tmp['fault_time'] = ft.loc[j, 'fault_time']
        tmp['tag'] = ft.loc[j, 'tag']
        # print(tmp)
        train_t.append(tmp)
train_ts = pd.concat(train_t)
train_ts.to_hdf(path + '/data/train_t/train_ts807.h5', 'df', mode='w')
print(len(train_ts))  # 2666
train_ts.head()


# In[ ]:


t = pd.read_hdf(path + '/data/train_t_h5/train_ts807.h5')
t.to_csv(path + '/data/train_t_csv/train_ts807.csv', index=False)

