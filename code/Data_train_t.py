#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
path = os.getcwd()


# In[5]:


train_tl = []
path_f = path + '/data/train_t/'
for file in os.listdir(path_f):
    print(file)
    tmp = pd.read_hdf(path_f + file)
    train_tl.append(tmp)
train_t = pd.concat(train_tl)
train_t.to_hdf(path + '/data/train_t.h5', 'df', mode='w')
train_t.to_csv(path + '/data/train_t.csv', index=False)
print(train_t.shape)  # (201975, 516)
train_t.head()


# In[21]:


train_t.describe()


# In[22]:


train_t.dtypes


# In[23]:


for col in train_t.columns:
    print(col + ' : ' + str(train_t[col].dtype))


# In[25]:


train_t.info()


# In[ ]:





# In[31]:


td = pd.read_csv(path + '/data/train_t.csv')
td.head()


# In[33]:


td['fault_time'] = pd.to_datetime(td['fault_time'])
td.head()


# In[35]:


td['dt'] = pd.to_datetime(td['dt'].map(str))
td.head()


# In[36]:


td.info()


# In[37]:


td.to_csv(path + '/data/train_t_time.csv', index=False)


# In[38]:


tt = pd.read_csv(path + '/data/train_t_time.csv')
tt.head()


# In[39]:


tt.info()


# In[40]:


tt['dt'] = pd.to_datetime(tt['dt'])
tt['fault_time'] = pd.to_datetime(tt['fault_time'])
tt.head()


# In[41]:


tt.info()


# In[42]:


tt['time_sub'] = tt['fault_time'] - tt['dt']
tt.head()


# In[43]:


tt.info()


# In[45]:


tt['time_subd'] = tt['time_sub'].map(lambda x:x.days)
tt.head()


# In[46]:


tt.info()


# In[56]:


tt1 = tt[(tt['time_subd'] <= 30) & (tt['time_subd'] >= 0)]
print(tt1.shape)  # (32547, 518)
tt1.head()


# In[57]:


tt1.to_csv(path + '/data/train_t/train_t_time30.csv', index=False)


# In[ ]:




