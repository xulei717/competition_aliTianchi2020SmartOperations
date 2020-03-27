#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from tqdm import tqdm
path = os.getcwd()
path


# In[12]:


# 提取正样本
train_t = []
ld = path + '/data/processed_date_data/'
for file in tqdm(os.listdir(ld)):
    if file[-3:] == 'csv' and file[:2] == '20':
        dt = pd.read_csv(ld + file)
        print(file)
        #print(dt.columns)
        dt = dt[dt['label']==1]
        print(dt.shape)
        train_t.append(dt)
tn = pd.concat(train_t, ignore_index=True)
print(tn.shape)  # (29992, 52)
tn.head()


# In[13]:


tn.to_csv(path + '/data/processed_date_data/traint.csv', index=False)


# In[14]:


traint = pd.read_csv(path + '/data/processed_date_data/traint.csv')
print(traint.shape)
traint.head()


# In[54]:


# 提取smart_5raw,smart_187raw,smart_188raw,smart_197raw,smart_198raw，五列任意等于0的正样本
traint0 = traint[(traint['smart_5raw']==0) | 
                (traint['smart_187raw']==0) | 
                (traint['smart_188raw']==0) | 
                (traint['smart_197raw']==0) | 
                (traint['smart_198raw']==0)
               ]
traint0.reset_index(drop=True, inplace=True)  # 一定要重新设定index
print(traint0.shape)  # (29497, 52)
traint0.tail(5)


# # kmeans聚类

# In[55]:


# 去除无关的列
cols = ['serial_number','model','manufacturer','dt','fault_time','differ_day','label']
raws = [x for x in traint0.columns if 'raw' in x]
print(len(cols), len(raws))  # 7 22
data = traint0[set(traint0.columns)-set(cols+raws)]
print(data.shape)  # (29497, 23)
data.head()


# In[56]:


# 用平均值填充空值
for column in list(data.columns[data.isnull().sum() > 0]):
    print(column)
    mean_val = data[column].mean()
    data[column].fillna(mean_val, inplace=True)
data.head()


# In[57]:


# 提取出值唯一的列
uq = []
for col in data.columns:
    un = data[col].unique()
    if len(un) == 1:
        print(col, un)
        uq.append(col)


# In[58]:


# 正样本中去除值为唯一的列
data = data[set(data.columns)-set(uq)]
print(data.shape)  # (29497, 19)
data.head()


# In[59]:


data.dtypes


# In[60]:


# kmeans聚类：设定簇数为10
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10, random_state=0).fit(data)


# In[61]:


# kmeans聚类后，每个正样本所属于的簇
print(len(kmeans.labels_))
kmeans.labels_


# In[62]:


# 获得kmeans聚类正样本后，簇的中心点
kmeans.cluster_centers_


# In[63]:


# 提取簇的下标值，即dataframe中的行数
import numpy as np
indexs_label0 = np.argwhere(kmeans.labels_==0)
indexs_label0 = [x[0] for x in indexs_label0]
print('indexs_label0: ', len(indexs_label0))  # 
indexs_label1 = np.argwhere(kmeans.labels_==1)
indexs_label1 = [x[0] for x in indexs_label1]
print('indexs_label1: ', len(indexs_label1))  # 
indexs_label2 = np.argwhere(kmeans.labels_==2)
indexs_label2 = [x[0] for x in indexs_label2]
print('indexs_label2: ', len(indexs_label2))  # 
indexs_label3 = np.argwhere(kmeans.labels_==3)
indexs_label3 = [x[0] for x in indexs_label3]
print('indexs_label3: ', len(indexs_label3))  # 
indexs_label4 = np.argwhere(kmeans.labels_==4)
indexs_label4 = [x[0] for x in indexs_label4]
print('indexs_label4: ', len(indexs_label4))  # 
indexs_label5 = np.argwhere(kmeans.labels_==5)
indexs_label5 = [x[0] for x in indexs_label5]
print('indexs_label5: ', len(indexs_label5))  # 
indexs_label6 = np.argwhere(kmeans.labels_==6)
indexs_label6 = [x[0] for x in indexs_label6]
print('indexs_label6: ', len(indexs_label6))  # 
indexs_label7 = np.argwhere(kmeans.labels_==7)
indexs_label7 = [x[0] for x in indexs_label7]
print('indexs_label7: ', len(indexs_label7))  # 
indexs_label8 = np.argwhere(kmeans.labels_==8)
indexs_label8 = [x[0] for x in indexs_label8]
print('indexs_label8: ', len(indexs_label8))  # 
indexs_label9 = np.argwhere(kmeans.labels_==9)
indexs_label9 = [x[0] for x in indexs_label9]
print('indexs_label9: ', len(indexs_label9))  # 
print(len(indexs_label0) + len(indexs_label1) + len(indexs_label2) + len(indexs_label3) + 
     len(indexs_label4) + len(indexs_label5) + len(indexs_label6) + len(indexs_label7) + 
     len(indexs_label8) + len(indexs_label9))


# In[65]:


# kmeans聚类获得的每个正样本所属的簇赋值给数据的label列
data.loc[indexs_label0,'kmeans_label'] = 0
data.loc[indexs_label1,'kmeans_label'] = 1
data.loc[indexs_label2,'kmeans_label'] = 2
data.loc[indexs_label3,'kmeans_label'] = 3
data.loc[indexs_label4,'kmeans_label'] = 4
data.loc[indexs_label5,'kmeans_label'] = 5
data.loc[indexs_label6,'kmeans_label'] = 6
data.loc[indexs_label7,'kmeans_label'] = 7
data.loc[indexs_label8,'kmeans_label'] = 8
data.loc[indexs_label9,'kmeans_label'] = 9
print(data.shape)
data.head()


# In[66]:


data.columns


# In[67]:


# plotly-express画散点图
import plotly.express as px
fig = px.scatter(data, x=data.index, y="smart_198_normalized", color="kmeans_label")
fig.show()  # kmeans_html


# In[69]:


# 求每个簇的中心点和最近点，最远点
# 簇的中心点
centers = kmeans.cluster_centers_
print(len(centers[0]), centers[0])


# In[71]:


# 欧式距离： np.linalg.norm
cl0 = data.loc[indexs_label0,:]
print(cl0.shape)
cl0.head()


# In[77]:


tp = np.array(cl0.iloc[0, :])[:-1] - np.array(centers[0])
print(len(tp))
print(np.linalg.norm(tp))
tp


# In[86]:


print(len(data.columns))
data.columns


# In[91]:


# 每个簇中离中心点最小和最大的欧式距离值
def cl_min_max(cl, cl_rt, col_index):
    cl, cl_min, cl_max  = cl, float('inf'), 0 
    indexs_label_kv = {'0':indexs_label0, '1':indexs_label1, '2':indexs_label2, '3':indexs_label3, '4':indexs_label4
                      , '5':indexs_label5, '6':indexs_label6, '7':indexs_label7, '8':indexs_label8, '9':indexs_label9}
    indexs_label = indexs_label_kv[str(cl)]
    center = centers[cl]
    #print(len(indexs_label), indexs_label)
    for i in tqdm(indexs_label):
        #tp = np.linalg.norm(np.array(data.iloc[i, :])[:-1] - np.array(center))
        tp = np.sqrt(abs(data.iloc[i,col_index]**2-center[col_index]**2))
        if tp < cl_min:
            cl_min = tp
        elif tp > cl_max:
            cl_max = tp
    cl_rt[str(cl)] = (cl_min, cl_max)
    return cl_rt


# In[93]:


def clrt(cluster_number, col_index):
    cl_rt = {}  # {簇数:(簇中欧式距离最小值，簇中欧式距离最大值)}
    for c in range(cluster_number):
        cl_rt = cl_min_max(c, cl_rt, col_index)
    return cl_rt


# In[95]:


col_cl = {}
cols = list(data.columns)
for c in range(len(cols)-1):
    col_cl[cols[c]] = clrt(10, c)
col_cl


# In[96]:


col_cl


# In[97]:


col_cl.keys()


# In[103]:


# 画每个特征的欧式距离最小值和最大值的柱状图
import plotly.express as px
a = pd.DataFrame.from_dict(col_cl['smart_198_normalized'],orient='index',columns=['min','max'])
#fig = px.bar(a, x=a.index, y=['min','max'])


# In[106]:


import random
import plotly as py
import plotly.graph_objs as go
#import plotly.graph_objs.layout.Legend as lgd
pyplt = py.offline.plot

a = pd.DataFrame.from_dict(col_cl['smart_198_normalized'],orient='index',columns=['min','max'])
print(a.head(1))

trace0 = go.Bar(
    x = a.index,
    y = a['min'],
    name = 'min',
)

trace1 = go.Bar(
    x = a.index,
    y = a['max'],
    name = 'max',
)

trace = [trace0,trace1]
layout = go.Layout(
    title = dict(text='smart_198_normalized',x=0.5,y=0.93),
    showlegend = True,
    legend = dict(x=1, y=0.5),
)
fig = go.Figure(data=trace, layout=layout)
pyplt(fig, filename=path + '/data/kmeans10_bar_html/smart_198_normalized.html')


# #  求每个特征在每个簇上的最小值和最大点

# In[111]:


ccmm = {}
indexs_label_kv = {'0':indexs_label0, '1':indexs_label1, '2':indexs_label2, '3':indexs_label3, '4':indexs_label4
                      , '5':indexs_label5, '6':indexs_label6, '7':indexs_label7, '8':indexs_label8, '9':indexs_label9}

def get_ccrt(custer_number, col):    
    ccrt = {}
    for cl in range(custer_number):
        indexs_label = indexs_label_kv[str(cl)]
        cc = list(data.loc[indexs_label,col])
        cc_min, cc_max = min(cc), max(cc)
        ccrt[str(cl)] = (cc_min, cc_max)
    
    return ccrt

for col in list(data.columns)[:-1]:
    ccmm[col] = get_ccrt(10, col)
ccmm


# In[113]:


data.columns


# In[146]:


# 画每个特征在每个簇上最小值和最大值的柱状图
import random
import plotly as py
import plotly.graph_objs as go
#import plotly.graph_objs.layout.Legend as lgd
pyplt = py.offline.plot
col_name = 'smart_189_normalized'
mm = pd.DataFrame.from_dict(ccmm[col_name],orient='index',columns=['min','max'])
print(mm.head(1))

trace0 = go.Bar(
    x = mm.index,
    y = mm['min'],
    name = 'min',
)

trace1 = go.Bar(
    x = mm.index,
    y = mm['max'],
    name = 'max',
)

trace = [trace0,trace1]
layout = go.Layout(
    title = dict(text=col_name,x=0.5,y=0.93),
    showlegend = True,
    legend = dict(x=1, y=0.5),
)
fig = go.Figure(data=trace, layout=layout)
pyplt(fig, filename=path + '/data/kmeans10_barMinMax_html/'+col_name+'.html')


# # 去掉10，240，241，242的raw和normalized列，除了3，别的列都只留raw，并把raw列的值转换到[0，100]，再聚类

# In[208]:


dt = pd.read_csv(path + '/data/processed_date_data/traint.csv')
print(dt.shape)
dt.tail()


# In[168]:


# 提取smart_5raw,smart_187raw,smart_188raw,smart_197raw,smart_198raw，五列任意等于0的正样本
dt0 = dt[(dt['smart_5raw']==0) | 
                (dt['smart_187raw']==0) | 
                (dt['smart_188raw']==0) | 
                (dt['smart_197raw']==0) | 
                (dt['smart_198raw']==0)
               ]
dt0.reset_index(drop=True, inplace=True)  # 一定要重新设定index
print(dt0.shape)  # (29497, 52)
dt0.tail(5)


# In[209]:


dt1 = dt[(dt['smart_5raw']==0) & 
                (dt['smart_187raw']==0) & 
                (dt['smart_188raw']==0) & 
                (dt['smart_197raw']==0) & 
                (dt['smart_198raw']==0)
               ]
dt1.reset_index(drop=True, inplace=True)  # 一定要重新设定index
print(dt1.shape)  # (29497, 52)
dt1.tail(5)


# In[210]:


# 去除无关的列
cols = ['serial_number','model','manufacturer','dt','fault_time','differ_day','label',
        'smart_10raw','smart_240raw','smart_241raw','smart_242raw',
        'smart_5raw','smart_187raw','smart_188raw','smart_197raw','smart_198raw']
raws = [x for x in dt1.columns if 'normalized' in x]
print(len(cols), len(raws))  # 16 23
data = dt1[list(set(dt1.columns) - set(cols+raws)) + ['smart_3_normalized']]
print(data.shape)  # (18976, 14)
data.head()


# In[211]:


data.describe()


# In[212]:


# 把最大值大于100的列的值转换到[0,100]
for col in list(data.columns):
    mi, mx = min(data[col]), max(data[col])
    if mx > 100:
        data[col] = data[col].map(lambda x: (x-mi)*100/mx)
    else:
        print(col, mi, mx)
data.describe()    


# In[213]:


# 用平均值填充空值
for column in list(data.columns[data.isnull().sum() > 0]):
    print(column)
    mean_val = data[column].mean()
    data[column].fillna(mean_val, inplace=True)
data.head()


# In[214]:


# 提取出值唯一的列
uq = []
for col in data.columns:
    un = data[col].unique()
    if len(un) == 1:
        print(col, un)
        uq.append(col)


# In[249]:


# kmeans聚类：设定簇数为3
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(data)


# In[250]:


# kmeans聚类后，每个正样本所属于的簇
print(len(kmeans.labels_))
kmeans.labels_


# In[251]:


# 提取簇的下标值，即dataframe中的行数
import numpy as np
indexs_label0 = np.argwhere(kmeans.labels_==0)
indexs_label0 = [x[0] for x in indexs_label0]
print('indexs_label0: ', len(indexs_label0))
indexs_label1 = np.argwhere(kmeans.labels_==1)
indexs_label1 = [x[0] for x in indexs_label1]
print('indexs_label1: ', len(indexs_label1))
indexs_label2 = np.argwhere(kmeans.labels_==2)
indexs_label2 = [x[0] for x in indexs_label2]
print('indexs_label2: ', len(indexs_label2))
# 2: 9692, 9282
# 3: 8328,4703,5945


# In[252]:


# kmeans聚类获得的每个正样本所属的簇赋值给数据的label列
data.loc[indexs_label0,'kmeans_label'] = 0
data.loc[indexs_label1,'kmeans_label'] = 1
data.loc[indexs_label2,'kmeans_label'] = 2
print(data.shape)
data.head()


# In[253]:


#  求每个特征在每个簇上的最小值和最大点
ccmm = {}
indexs_label_kv = {'0':indexs_label0, '1':indexs_label1, '2':indexs_label2}
custer_number = 3
def get_ccrt(custer_number, col):    
    ccrt = {}
    for cl in range(custer_number):
        indexs_label = indexs_label_kv[str(cl)]
        cc = list(data.loc[indexs_label,col])
        cc_min, cc_max = min(cc), max(cc)
        ccrt[str(cl)] = (cc_min, cc_max)
    
    return ccrt

for col in list(data.columns)[:-1]:
    ccmm[col] = get_ccrt(custer_number, col)
ccmm


# In[254]:


data.columns


# In[256]:


# 画每个特征在每个簇上最小值和最大值的柱状图
import random
import plotly as py
import plotly.graph_objs as go
#import plotly.graph_objs.layout.Legend as lgd
pyplt = py.offline.plot
os_list = 'kmeans3_barMinMax_html'
col_name = 'smart_1raw'
mm = pd.DataFrame.from_dict(ccmm[col_name],orient='index',columns=['min','max'])
print(mm.head(1))

trace0 = go.Bar(
    x = mm.index,
    y = mm['min'],
    name = 'min',
)

trace1 = go.Bar(
    x = mm.index,
    y = mm['max'],
    name = 'max',
)

trace = [trace0,trace1]
layout = go.Layout(
    title = dict(text=col_name,x=0.5,y=0.93),
    showlegend = True,
    legend = dict(x=1, y=0.5),
)
fig = go.Figure(data=trace, layout=layout)
pyplt(fig, filename=path + '/data/'+os_list+'/'+col_name+'.html')


# In[ ]:




