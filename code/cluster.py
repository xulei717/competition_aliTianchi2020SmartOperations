#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import pandas as pd
path = os.getcwd()
path


# In[6]:


# 读取正样本
train_t = pd.read_csv(path + '/data/train_t/train_t_time30_52.csv')
print(train_t.shape)
train_t.head()


# In[7]:


# 去除无关的列
cols = ['serial_number','model','manufacturer','dt','fault_time','tag','time_subd']
raws = [x for x in train_t.columns if 'raw' in x]
print(len(raws), raws)
data = train_t[set(train_t.columns)-set(cols+raws)]
print(data.shape)
data.head()


# In[8]:


# 用平均值填充空值
for column in list(data.columns[data.isnull().sum() > 0]):
    mean_val = data[column].mean()
    data[column].fillna(mean_val, inplace=True)
data.head()


# In[15]:


data.describe()


# In[9]:


# 提取出值唯一的列
for col in data.columns:
    un = data[col].unique()
    if len(un) == 1:
        print(col, un)


# In[10]:


# 正样本中去除值为唯一的列
data = data[set(data.columns)-set(['smart_241_normalized', 'smart_240_normalized','smart_242_normalized','smart_10_normalized'])]
print(data.shape)
data.head()


# In[18]:


data.dtypes


# # kmeans聚类

# In[11]:


# kmeans聚类：设定簇为2
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(data)


# In[12]:


# kmeans聚类后，每个正样本所属于的簇：0簇或1簇
print(len(kmeans.labels_))
kmeans.labels_


# In[22]:


# 获得kmeans聚类正样本后，两个簇的中心点
kmeans.cluster_centers_


# In[13]:


# 提取0簇和1簇的下标值，即dataframe中的行数
import numpy as np
indexs_label1 = np.argwhere(kmeans.labels_==1)
indexs_label1 = [x[0] for x in indexs_label1]
print(len(indexs_label1), indexs_label1)  # 15831
indexs_label0 = np.argwhere(kmeans.labels_==0)
indexs_label0 = [x[0] for x in indexs_label0]
print(len(indexs_label0), indexs_label0)  # 16716


# In[14]:


# kmeans聚类获得的每个正样本所属的簇赋值给数据的label列
data.loc[indexs_label1,'label'] = 1
data.loc[indexs_label0,'label'] = 0
print(data.shape)
data.head()


# In[17]:


# plotly画散点图
import random
import plotly as py
import plotly.graph_objs as go
#import plotly.graph_objs.layout.Legend as lgd
pyplt = py.offline.plot

trace0 = go.Scatter(
    x = data.loc[indexs_label0, 'smart_1_normalized'],
    y = data.loc[indexs_label0, 'smart_197_normalized'],
    name = 'label0',
)

trace1 = go.Scatter(
    x = data.loc[indexs_label1, 'smart_1_normalized'],
    y = data.loc[indexs_label1, 'smart_197_normalized'],
    name = 'label1',
)

trace = [trace0,trace1]
layout = go.Layout(
    title = dict(text='smart_1_normalized',x=0.5,y=0.93),
    showlegend = True,
    legend = dict(x=1, y=0.5),
)
fig = go.Figure(data=trace, layout=layout)
pyplt(fig, filename=path + '/data/kmeans_html/smart_1_normalized.html')


# In[19]:


data.columns


# In[43]:


# plotly-express画散点图
import plotly.express as px
fig = px.scatter(data, x=data.index, y="smart_9_normalized", color="label")
fig.show()  # kmeans_html


# In[ ]:


kmeans:195,189,9,5


# # 层次聚类

# In[45]:


# 正样本训练层次聚类：设定为两类
from sklearn.cluster import AgglomerativeClustering
colsf = ['smart_199_normalized', 'smart_187_normalized', 'smart_3_normalized',
       'smart_184_normalized', 'smart_197_normalized', 'smart_7_normalized',
       'smart_188_normalized', 'smart_193_normalized', 'smart_190_normalized',
       'smart_4_normalized', 'smart_195_normalized', 'smart_5_normalized',
       'smart_192_normalized', 'smart_12_normalized', 'smart_194_normalized',
       'smart_189_normalized', 'smart_1_normalized', 'smart_198_normalized',
       'smart_9_normalized']
agg = AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='average').fit(data[colsf])
agg


# In[46]:


# 层次聚类结果中每个样本所属的簇，分为0簇和1簇两类
print(agg.labels_)
agg.labels_


# In[47]:


# 提取0簇和1簇的下标值，即dataframe中的行数
import numpy as np
indexs_label11 = np.argwhere(agg.labels_==1)
indexs_label11 = [x[0] for x in indexs_label11]
print(len(indexs_label11), indexs_label11)  # 32450
indexs_label00 = np.argwhere(agg.labels_==0)
indexs_label00 = [x[0] for x in indexs_label00]
print(len(indexs_label00), indexs_label00)  # 97


# In[38]:


# 层次聚类获得的0簇与kmeans聚类获得的0簇和1簇的差集
indexs_label00_0 = list(set(indexs_label00)-set(indexs_label0))  # 97-16716
print(len(indexs_label00_0), indexs_label00_0)  # 0
indexs_label00_1 = list(set(indexs_label00)-set(indexs_label1))  # 97-15831
print(len(indexs_label00_1), indexs_label00_1)  # 97


# In[48]:


# kmeans聚类获得的每个正样本所属的簇赋值给数据的label列
dtf = data[colsf]
dtf.loc[indexs_label11,'label'] = 1
dtf.loc[indexs_label00,'label'] = 0
print(dtf.shape)
data.head()


# In[49]:


dtf.columns


# In[68]:


# plotly-express画散点图
import plotly.express as px
fig = px.scatter(dtf, x=dtf.index, y="smart_9_normalized", color="label")
fig.show()  # agg_html


# In[ ]:


agg: 197,190,5,194,198,


# # 用层次聚类负样本，并从每个簇中提取样本

# In[4]:


# 读取负样本-201807
disk807 = pd.read_csv(path + '/data/disk_sample_smart_log_2018_Q2/disk_sample_smart_log_201807.csv')
print(disk807.shape)  # (4855732, 515)
disk807.head()


# In[6]:


# 从负样本取出与正样本一样的列
cols0 = ['smart_197_normalized','smart_1_normalized', 'smart_12_normalized', 
         'smart_190_normalized', 'smart_195_normalized','smart_199_normalized',
'smart_4_normalized','smart_198_normalized','smart_184_normalized','smart_188_normalized', 
'smart_7_normalized','smart_187_normalized','smart_5_normalized','smart_3_normalized',
'smart_192_normalized','smart_194_normalized','smart_9_normalized','smart_193_normalized',
'smart_189_normalized']
dataf = disk807[cols0]
print(dataf.shape)  # (4855732, 19)
dataf.head()


# In[7]:


# 用平均值填充空值
for column in list(dataf.columns[dataf.isnull().sum() > 0]):
    mean_val = dataf[column].mean()
    dataf[column].fillna(mean_val, inplace=True)
dataf.head()


# In[8]:


# 提取出值唯一的列
cols_uq = []
for col in dataf.columns:
    un = dataf[col].unique()
    if len(un) == 1:
        cols_uq.append(col)
print(len(cols_uq), cols_uq)


# In[9]:


dataf.to_csv(path + '/data/disk_sample_smart_log_2018_Q2/disk_sample_smart_log_201807_19.csv', index=False)


# In[2]:


dataf = pd.read_csv(path + '/data/disk_sample_smart_log_2018_Q2/disk_sample_smart_log_201807_19.csv')


# In[ ]:


# 用层次聚类把负样本分为20个簇,10/20个簇服务为挂
from sklearn.cluster import AgglomerativeClustering
aggf = AgglomerativeClustering(n_clusters=100,affinity='euclidean',linkage='average').fit(dataf)
aggf


# In[4]:


# kmeans聚类：设定簇为100
from sklearn.cluster import KMeans
kmeansf = KMeans(n_clusters=100, random_state=0).fit(dataf)


# In[ ]:





# In[ ]:




