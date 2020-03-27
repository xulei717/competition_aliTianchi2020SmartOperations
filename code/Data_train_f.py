#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os 
import array
import numpy as np
import pandas as pd
from tqdm import tqdm
path = os.getcwd()


# In[5]:


test = pd.read_csv(path + '/data/disk_sample_smart_log_test_a.csv')
print(test.shape)
test.head()


# In[6]:


test = test.sort_values(['serial_number','dt'])
test = test.drop_duplicates().reset_index(drop=True)
print(test.shape)
test.head()


# In[7]:


## test中nunique为1且没空值的特征
nuiq_list =[]
for i in tqdm([col for col in test.columns if col not in ['manufacturer','model']]):
    if (test[i].nunique() == 1)&(test[i].isnull().sum() == 0):
        nuiq_list.append(i)
print(len(nuiq_list))
nuiq_list


# In[8]:


## test中全为空的列
df= pd.DataFrame()
df['fea'] = test.isnull().sum().index
df['isnull_sum'] = test.isnull().sum().values
kong_list = df.loc[df.isnull_sum == test.shape[0]]['fea'].values
print(len(kong_list))


# In[27]:


print(kong_list[:5])


# In[9]:


## 去掉test中值唯一且没有空值的列和全为空值的列
fea_list = list(set(test.columns)-set(kong_list) - set(nuiq_list))
print(len(fea_list), test.shape[1], '465')


# In[31]:


test1 = test[fea_list]
test1.to_csv(path + '/data/test/test49.csv', index=False)


# In[ ]:





# In[32]:


train_t = pd.read_csv(path + '/data/train_t/train_t_time30.csv')
print(train_t.shape)  # (32547, 518)
train_t.head()


# In[35]:


train_t1 = train_t[test1.columns]
print(train_t1.shape)
train_t1.head()  # (32547, 49)


# In[37]:


train_t1.to_csv(path + '/data/train_t/train_t_time30_49.csv', index=False)


# In[ ]:





# In[10]:


# ## 构建训练集的负样本
def trainf(path_dt,path_tt,path_csv):
    ## 初始的训练数据 
    dt = pd.read_csv(path + '/data' + path_dt)
    dt1 = dt.iloc[:, 1:]
    print(dt1.shape)  # (1610007, 514)
    ## 与tag匹配的训练数据
    tt = pd.read_csv(path + '/data' + path_tt)
    print(tt.shape)  # (11791, 516)
    ## 构建训练集的负样本
    dt2 = dt1[fea_list]
    print(dt2.shape)  # (1610007, 49)
    tt1 = tt[fea_list]
    print(tt1.shape)  # (11791, 49)

    mms = tt1[["manufacturer","model","serial_number"]].values
    dtv = dt2[["manufacturer","model","serial_number"]].values
    c = np.isin(dtv, mms)
    indexs_t = []
    for i in range(len(c)):
        if False not in c[i]:
            indexs_t.append(i)
    print(len(indexs_t))  # 235
    train_f = dt2.drop(indexs_t, axis=0)
    train_f.to_csv(path + '/data/train_f' + path_csv, index=False)
    print(train_f.shape)  # (1595488, 49)
    train_f.head()


# In[ ]:


path_dt = '/disk_sample_smart_log_2017/disk_sample_smart_log_201708.csv'
path_tt = '/train_t_csv/train_ts708.csv'
path_csv = '/train_f_708.csv'
trainf(path_dt,path_tt,path_csv)


# In[ ]:




