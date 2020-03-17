#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
path = os.getcwd()
path


# In[3]:


train_t = pd.read_csv(path + '/data/train_t/train_t_time30_49.csv')
print(train_t.shape)
train_t.head()


# In[4]:


train_t.columns


# In[31]:


train_all = pd.read_csv(path + '/data/train_t/train_t_time30.csv')
print(train_all.shape)
train_all


# In[32]:


train = train_all[train_all['time_subd']<=30][list(train_t.columns)+['fault_time','tag','time_subd']]
print(train.shape)
train.head()


# In[33]:


cols = list(train.columns)
cols_lf = ['serial_number','model','manufacturer','dt','fault_time','tag','time_subd']
cols1 = cols_lf + sorted(set(cols)-set(cols_lf))
train = train[cols1]
print(train.shape)
train.head()


# In[34]:


train.to_csv(path + '/data/train_t/train_t_time30_52.csv', index=False)


# In[5]:


import plotly.express as px
df = px.data.gapminder().query("country=='Canada'")
fig = px.line(df, x="year", y="lifeExp", title='Life expectancy in Canada')
fig.show()


# In[6]:


df


# In[35]:


train.groupby(['serial_number','model']).size()


# In[37]:


train.groupby(['serial_number','model']).size().sort_values(ascending=False)


# In[40]:


train


# In[46]:


train.dtypes


# In[76]:


disk, model, col = 'disk_99991', 1, 'smart_7raw'
dt = train[(train['serial_number']==disk) & (train['model']==model)]
fig1 = px.line(dt[['dt',col]].sort_values(by=['dt'],axis=0,ascending=[True]) , x='dt', y=col, title=disk +': '+ col + ' + ' + str(model))
fig1.show()


# In[81]:


import math
disk, model, col = 'disk_99991', 1, 'smart_5raw'
dt = train[(train['model']==model)]
dt[col] = dt[col].apply(lambda x:math.log1p(x))
fig1 = px.line(dt[['dt',col,'serial_number']].sort_values(by=['dt'],axis=0,ascending=[True]) , x='dt', y=col,color='serial_number', title=col + ' + ' + str(model))
fig1.show()


# In[55]:


dt = train_all[(train_all['serial_number']=='disk_99991') & (train_all['model']==1)]
fig1 = px.line(dt[['dt','smart_1raw']].sort_values(by=['dt'],axis=0,ascending=[True]) , x='dt', y='smart_1raw', title='smart_1raw: disk_99991 + 1')
fig1.show()


# In[67]:


import math

dt = train_all[(train_all['serial_number']=='disk_99991') & (train_all['model']==1)][['smart_1raw']]
#dt['smart'] = dt['smart_1raw'].apply(lambda x:int(x)//10000000)
dt['smart'] = dt['smart_1raw'].apply(lambda x:int(math.log(x)))
dt = dt.groupby(['smart']).size()
dtd = {'smart_1raw':dt.index, 'num':dt.values}
dtdt = pd.DataFrame(dtd)
print(dtdt.head())
fig1 = px.line(dtdt, x='smart_1raw', y='num', title='smart_1raw: num')
fig1.show()


# In[ ]:





# In[10]:


fig1 = px.line(train_t[train_t['serial_number']=='disk_143888'], x='dt', y='smart_1raw', title='smart_1raw: disk_143888')
fig1.show()


# In[ ]:




