#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import numpy as np
from tqdm import tqdm
path = os.getcwd()
path


# In[3]:


test = pd.read_csv(path + '/data/processed_date_data/test.csv')
print(test.shape)  # (178028, 49)
test1 = test[(test['smart_5raw'] > 0) | 
                 (test['smart_187raw'] > 0) | 
                 (test['smart_188raw'] > 0) | 
                 (test['smart_197raw'] > 0) | 
                 (test['smart_198raw'] > 0)]  # 筛选raw:5,187,188,197,198任一大于0
print(test1.shape)  # (33111, 49)
tg = test1.groupby(["manufacturer", "model", "serial_number"])['dt'].max().reset_index()
print(tg.shape)  # 2213
tg.head()


# In[6]:


test = pd.read_csv(path + '/data/processed_date_data/test.csv')
print(test.shape)  # (178028, 49)
test1 = test[(test['smart_5_normalized'] < 100) | 
                 (test['smart_187_normalized'] < 100) | 
                 (test['smart_188_normalized'] < 100) | 
                 (test['smart_197_normalized'] < 100) | 
                 (test['smart_198_normalized'] < 100)]  # 筛选raw:5,187,188,197,198任一大于0
print(test1.shape)  # 
tg = test1.groupby(["manufacturer", "model", "serial_number"])['dt'].max().reset_index()
print(tg.shape)  # 
tg.head()


# In[7]:


# 把最终结果里的dt转换成test里最晚出现的时间
test = pd.read_csv(path + '/data/processed_date_data/test.csv')
td = test[test['model']==1]
td = td[['serial_number','dt']].groupby(['serial_number'])['dt'].max().reset_index()
print(td.shape)
tr = tg.merge(td, how='left', on='serial_number')
print(tr.shape)
tr['dt'] = tr['dt_y']
del tr['dt_x']
del tr['dt_y']
tr.drop_duplicates(inplace=True)
tr.reset_index(drop=True, inplace=True)
tr.to_csv(path + '/result/result_rules_max_test141.csv', index=False, header=None)
print(tr.shape)
tr.head()


# In[ ]:




