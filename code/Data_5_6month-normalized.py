#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import numpy as np
from tqdm import tqdm
path = os.getcwd()
path


# # 只用5 6月份的数据 然后只保留正常盘 正负样本都是 然后负样本只保留每块盘的最后一天 然后看看正负样本的比例

# In[ ]:


# 正样本2018-5/6月正常盘数据：2931
# 负样本2018-5/6月正常盘数据：96977
# 测试集数据： 176108
(2931, 41)
(96977, 41)
(176108, 38)


# In[ ]:


# 特征上 首先10，240，241，242还是要去掉
# 然后对1，5，7，199的raw 计算其跟前一天的差值（如果前一天没有记录就计算跟上一条记录的差值除以相隔天数）
# 然后 去掉1，195的raw 和 5，187，188，194，197，198，199的normalized（其中5，187，188，197，198因为都是正常盘值都是100所以去掉）
# days还是保留

# 特征处理完以后就训练正常盘的模型 然后预测test里的正常盘，看一下预测出来正常盘有多少为1


# In[91]:


# 提取测试集的正常样本
test = pd.read_csv(path + '/data/processed_date_data/test.csv')
print(test.shape)
test.head()


# In[92]:


# 对1，5，7，199的raw 计算其跟前一天的差值（如果前一天没有记录就计算跟上一条记录的差值除以相隔天数）
t1 = test.sort_values(by=['serial_number','dt'], ascending=True).reset_index(drop=True)
print(t1.shape)
t1.head()


# In[94]:


t1[['smart_1raw_dtdiff','smart_5raw_dtdiff','smart_7raw_dtdiff','smart_199raw_dtdiff']] = t1[['smart_1raw','smart_5raw','smart_7raw','smart_199raw']].diff().fillna(0)[['smart_1raw','smart_5raw','smart_7raw','smart_199raw']]
t1.shape


# In[95]:


t1[['smart_1raw_dtdiff','smart_5raw_dtdiff','smart_7raw_dtdiff','smart_199raw_dtdiff','smart_1raw','smart_5raw','smart_7raw','smart_199raw']].head()


# In[5]:


def func_dtdiff(x, index, col):
    if index == 0 or x.loc[index-1,'serial_number'] != x.loc[index, 'serial_number']:
        return 0
    diff = (x.loc[index, col] - x.loc[index-1,col])/((
            pd.to_datetime(x.loc[index,'dt'])-pd.to_datetime(x.loc[index-1,'dt'])).days)
    return diff


# In[31]:


# 对1，5，7，199的raw 计算其跟前一天的差值（如果前一天没有记录就计算跟上一条记录的差值除以相隔天数）
t1 = test.sort_values(by=['serial_number','dt'], ascending=True).reset_index()
# 不应该用for循环，可以使用diff实现
for i in tqdm(range(t1.shape[0])):
    t1.loc[i, 'smart_1raw_dtdiff'] = func_dtdiff(t1, i, 'smart_1raw')
    t1.loc[i, 'smart_5raw_dtdiff'] = func_dtdiff(t1, i, 'smart_5raw')
    t1.loc[i, 'smart_7raw_dtdiff'] = func_dtdiff(t1, i, 'smart_7raw')
    t1.loc[i, 'smart_199raw_dtdiff'] = func_dtdiff(t1, i, 'smart_199raw')
print(t1.shape)
t1[['serial_number','dt','smart_1raw','smart_1raw_dtdiff'
    ,'smart_5raw','smart_5raw_dtdiff'
    ,'smart_7raw','smart_7raw_dtdiff'
   ,'smart_199raw','smart_199raw_dtdiff']].head(1000)


# In[7]:


t1.head()


# In[32]:


# 筛选正常样本
test1 = t1[(t1['smart_5_normalized'] == 100) & 
                 (t1['smart_187_normalized'] == 100) & 
                 (t1['smart_188_normalized'] == 100) & 
                 (t1['smart_197_normalized'] == 100) & 
                 (t1['smart_198_normalized'] == 100)]  # 筛选raw:5,187,188,197,198任一大于0
print(test1.shape)  # 


# In[33]:


# 删除测试集中某些列
del_cols = ['smart_10raw','smart_10_normalized','smart_240raw','smart_240_normalized',
           'smart_241raw','smart_241_normalized','smart_242raw','smart_242_normalized',
           'smart_1raw','smart_195raw',
           'smart_5_normalized','smart_187_normalized','smart_188_normalized','smart_194_normalized',
           'smart_197_normalized','smart_198_normalized','smart_199_normalized','index']
test2 = test1[set(test1.columns)-set(del_cols)]
print(test2.shape)
#print(test.columns)
test2.head(2)


# In[34]:


test2.to_csv(path + '/data/processed_date_data/test_month5_6.csv', index=False)


# In[35]:


# 提取正样本的2018-5/6月数据
traint = pd.read_csv(path + '/data/processed_date_data/traint.csv')
print(traint.shape)  # (29992, 52)
traint = traint[traint['model']==1]
print(traint.shape)
traint['dt'] = pd.to_datetime(traint['dt'])
traint = traint[(traint['dt'].apply(lambda x:x.year==2018)) & (traint['dt'].apply(lambda x:x.month in [5,6]))]
print(traint.shape)
traint.head(2)


# In[36]:


# 对1，5，7，199的raw 计算其跟前一天的差值（如果前一天没有记录就计算跟上一条记录的差值除以相隔天数）
tt1 = traint.sort_values(by=['serial_number','dt'], ascending=True).reset_index()

for i in range(tt1.shape[0]):
    tt1.loc[i, 'smart_1raw_dtdiff'] = func_dtdiff(tt1, i, 'smart_1raw')
    tt1.loc[i, 'smart_5raw_dtdiff'] = func_dtdiff(tt1, i, 'smart_5raw')
    tt1.loc[i, 'smart_7raw_dtdiff'] = func_dtdiff(tt1, i, 'smart_7raw')
    tt1.loc[i, 'smart_199raw_dtdiff'] = func_dtdiff(tt1, i, 'smart_199raw')
print(tt1.shape)
tt1[['serial_number','dt','smart_1raw','smart_1raw_dtdiff'
    ,'smart_5raw','smart_5raw_dtdiff'
    ,'smart_7raw','smart_7raw_dtdiff'
   ,'smart_199raw','smart_199raw_dtdiff']].head(1000)


# In[13]:


tt1.head()


# In[37]:


# 提取正样本的正常样本
traint1 = tt1[(tt1['smart_5_normalized'] == 100) & 
                 (tt1['smart_187_normalized'] == 100) & 
                 (tt1['smart_188_normalized'] == 100) & 
                 (tt1['smart_197_normalized'] == 100) & 
                 (tt1['smart_198_normalized'] == 100)]  # 筛选raw:5,187,188,197,198任一大于0
print(traint1.shape)  # 


# In[38]:


# 删除正样本中某些列
del_cols = ['smart_10raw','smart_10_normalized','smart_240raw','smart_240_normalized',
           'smart_241raw','smart_241_normalized','smart_242raw','smart_242_normalized',
           'smart_1raw','smart_195raw',
           'smart_5_normalized','smart_187_normalized','smart_188_normalized','smart_194_normalized',
           'smart_197_normalized','smart_198_normalized','smart_199_normalized','index']
traint2 = traint1[set(traint1.columns)-set(del_cols)]
print(traint2.shape)
#print(traint.columns)
traint2.head(2)


# In[41]:


for column in list(traint2.columns[traint2.isnull().sum() > 0]):
    print(column)


# In[39]:


traint2.to_csv(path + '/data/processed_date_data/traint_month5_6.csv', index=False)


# In[42]:


# 提取负样本的2018-5/6月的正常数据
trainf = []
del_cols = ['smart_10raw','smart_10_normalized','smart_240raw','smart_240_normalized',
           'smart_241raw','smart_241_normalized','smart_242raw','smart_242_normalized']
path_files = path + '/data/processed_date_data/'
for file in os.listdir(path_files):
    if file == '201805.csv' or file == '201806.csv':
        tp = pd.read_csv(path_files+file)
        print(file, tp.shape)
        tp = tp[set(tp.columns)-set(del_cols)]  # 删除8个列：raw+normalized:10，240，241，242
        tp = tp[tp['model'] == 1]  # 筛选model==1
        tp = tp[tp['label'] == 0]  # 筛选出负样本
        tpp = tp[(tp['smart_5_normalized'] == 100) & 
                 (tp['smart_187_normalized'] == 100) & 
                 (tp['smart_188_normalized'] == 100) & 
                 (tp['smart_197_normalized'] == 100) & 
                 (tp['smart_198_normalized'] == 100)]  # 筛选raw:5,187,188,197,198任一大于0
        tpp.reset_index(drop=True, inplace=True)  # 一定要重新设定index
        print(tpp.shape)
        trainf.append(tpp)
tf = pd.concat(trainf, ignore_index=True)
print(tf.shape)
pre = tf.groupby(["manufacturer", "model", "serial_number"])['dt'].max().reset_index()
print(pre.shape)
pre.head(2)


# In[43]:


# 提取serial_number的dt最晚的数据，index
tf1 = tf.sort_values(by=['serial_number','dt'], ascending=False)
tfd = tf1.drop_duplicates(subset=["manufacturer", "model", "serial_number"], keep='first')
print(tfd.shape)
tfd.head()


# In[45]:


tf1.head()


# In[48]:


tf1.loc[list(tfd.index),:]


# In[44]:


# 提取serial_number的dt次晚的数据，index
print(tf1.shape, tfd.shape)
tf1=tf1.drop(labels=list(tfd.index),axis=0) 
print(tf1.shape)
tfdd = tf1.drop_duplicates(subset=["manufacturer", "model", "serial_number"], keep='first')
tfdd.head(2)


# In[45]:


tfdd.shape


# In[46]:


# serial_number的dt最晚的数据和serial_number的dt次晚的数据concat，
# 然后对1，5，7，199的raw 计算其跟前一天的差值（如果前一天没有记录就计算跟上一条记录的差值除以相隔天数）
tf1 = pd.concat([tfd, tfdd])
tf1 = tf1.reset_index()
print(tfd.shape, tfdd.shape, tf1.shape)
tf1.head(2)


# In[47]:


for i in tqdm(range(tf1.shape[0])):
    tf1.loc[i, 'smart_1raw_dtdiff'] = func_dtdiff(tf1, i, 'smart_1raw')
    tf1.loc[i, 'smart_5raw_dtdiff'] = func_dtdiff(tf1, i, 'smart_5raw')
    tf1.loc[i, 'smart_7raw_dtdiff'] = func_dtdiff(tf1, i, 'smart_7raw')
    tf1.loc[i, 'smart_199raw_dtdiff'] = func_dtdiff(tf1, i, 'smart_199raw')
print(tf1.shape)
tf1[['serial_number','dt','smart_1raw','smart_1raw_dtdiff'
    ,'smart_5raw','smart_5raw_dtdiff'
    ,'smart_7raw','smart_7raw_dtdiff'
   ,'smart_199raw','smart_199raw_dtdiff']].head(1000)


# In[23]:


tf1.head()


# In[48]:


# 提取serial_number的dt最晚的数据，index
tf2 = tf1.sort_values(by=['serial_number','dt'], ascending=False)
tfd2 = tf2.drop_duplicates(subset=["manufacturer", "model", "serial_number"], keep='first')
print(tfd2.shape)
tfd2.head()


# In[49]:


# 删除负样本中某些列
del_cols = ['smart_10raw','smart_10_normalized','smart_240raw','smart_240_normalized',
           'smart_241raw','smart_241_normalized','smart_242raw','smart_242_normalized',
           'smart_1raw','smart_195raw',
           'smart_5_normalized','smart_187_normalized','smart_188_normalized','smart_194_normalized',
           'smart_197_normalized','smart_198_normalized','smart_199_normalized','index']
trainf = tfd2[set(tfd2.columns)-set(del_cols)]
print(trainf.shape)
print(trainf.columns)
trainf.head(2)


# In[50]:


trainf.to_csv(path + '/data/processed_date_data/trainf_month5_6.csv', index=False)


# In[51]:


# 用disk_first.csv每个硬盘观察的第一天与样本每行匹配获得每个硬盘当时的观察天数：dt-disk_first.dt
data = pd.read_csv(path + '/data/processed_date_data/disk_first.csv')
print(data.shape)  # (174917, 41)
data = data[data['model']==1]
print(data.shape)  # (102630, 41)
dt = data[['serial_number','dt']]
dt['dt_first'] = dt['dt']
del dt['dt']
dt.head(2)


# In[52]:


# 正样本构建硬盘观察天数
tt = pd.read_csv(path + '/data/processed_date_data/traint_month5_6.csv')
print(tt.shape) 
tt = tt.merge(dt, how='left',on='serial_number')
tt['days'] = (pd.to_datetime(tt['dt'])-pd.to_datetime(tt['dt_first'])).dt.days
tt.to_csv(path + '/data/processed_date_data/traintd_month5_6.csv', index=False)
print(tt.shape) 
tt.head(2)


# In[53]:


for column in list(tt.columns[tt.isnull().sum() > 0]):
    print(column)


# In[54]:


# 负样本构建硬盘观察天数
tf = pd.read_csv(path + '/data/processed_date_data/trainf_month5_6.csv')
print(tf.shape)  # (7505, 44)
tf = tf.merge(dt, how='left',on='serial_number')
tf['days'] = (pd.to_datetime(tf['dt'])-pd.to_datetime(tf['dt_first'])).dt.days
tf.to_csv(path + '/data/processed_date_data/trainfd_month5_6.csv', index=False)
print(tf.shape)
tf.head()


# In[55]:


# 测试集构建硬盘观察天数
tf = pd.read_csv(path + '/data/processed_date_data/test_month5_6.csv')
print(tf.shape)  # (7505, 44)
tf = tf.merge(dt, how='left',on='serial_number')
tf['days'] = (pd.to_datetime(tf['dt'])-pd.to_datetime(tf['dt_first'])).dt.days
tf.to_csv(path + '/data/processed_date_data/testd_month5_6.csv', index=False)
print(tf.shape)
tf.head()


# # 用正样本的正常数据和负样本的正常数据预测测试集的正常数据

# In[76]:


# 获取正负样本，并预测测试集
tt0 = pd.read_csv(path + '/data/processed_date_data/traintd_month5_6.csv')
print(tt.shape)  
tf0 = pd.read_csv(path + '/data/processed_date_data/trainfd_month5_6.csv')
print(tf0.shape)  
test = pd.read_csv(path + '/data/processed_date_data/testd_month5_6.csv')
print(test.shape)  


# In[77]:


tf0 = tf0.sample(frac=0.25, random_state=0)  # 1:10
#tf0 = tf0.sample(frac=0.125, random_state=0)  # 1:5
#tf = tf.sample(frac=0.05, random_state=0)  # 1:2
#tf = tf.sample(frac=0.025, random_state=0)  # 1:1
print(tf0.shape)


# In[39]:


tf0.head()


# In[40]:


tt0.head()


# In[41]:


test.head()


# In[78]:


# 把正负样本合并一起
train = pd.concat([tt0, tf0])
print(train.shape)


# In[4]:


tf['dt'] = pd.to_datetime(tf['dt'])
tf5 = tf[(tf['dt'].apply(lambda x:x.year==2018)) & (tf['dt'].apply(lambda x:x.month==5))]
print(tf5.shape)
tf6 = tf[(tf['dt'].apply(lambda x:x.year==2018)) & (tf['dt'].apply(lambda x:x.month==6))]
print(tf6.shape)


# In[79]:


for column in list(tt0.columns[tt0.isnull().sum() > 0]):
    print(column)


# In[80]:


for column in list(tf0.columns[tf0.isnull().sum() > 0]):
    print(column)


# In[81]:


for column in list(train.columns[train.isnull().sum() > 0]):
    print(column)
    #mean_val = data[column].mean()
    #data[column].fillna(mean_val, inplace=True)
#data.head()


# In[88]:


for column in list(test.columns[test.isnull().sum() > 0]):
    print(column)


# In[83]:


features = [x for x in train.columns if x not in ["serial_number", "manufacturer", "model", "dt", "fault_time", "differ_day", "label", 'dt_first']]
print(train[features].shape, test[features].shape)
target = 'label'


# In[61]:


import sys
import math
import glob
import json
import pandas as pd 
import pandas_profiling
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
pd.options.display.width = 500
import numpy
import datetime
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, train_test_split, GroupKFold
from sklearn.metrics import roc_auc_score, mean_squared_error
import lightgbm
import catboost
from skopt import BayesSearchCV
from sklearn.model_selection import KFold
from sklearn.externals import joblib


# In[84]:


# lightgbm模型
fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)

X = train[features].copy()
y = train[target]
models = list()
pred = numpy.zeros(len(test))
oof = numpy.zeros(len(X))

for index, (train_idx, val_idx) in enumerate(fold.split(X, y)):
    train_X = X.iloc[train_idx]
    train_y = y.iloc[train_idx]
    val_X = X.iloc[val_idx]
    val_y = y.iloc[val_idx]
    model = lightgbm.LGBMClassifier(
        boosting_type="gbdt",
        #metric='auc',
        learning_rate=0.001,
        n_estimators=5000,
        num_leaves=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=2020)
    model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=500)
    models.append(model)
    val_pred = model.predict(val_X)
    oof[val_idx] = val_pred
    val_f1 = metrics.f1_score(val_y, val_pred)
    print(index, 'val f1', val_f1)
    joblib.dump(model, path + '/models/model_month56_0.25_' + str(val_f1) + '.txt')

    test_pred = model.predict(test[features])
    pred += test_pred / 5

print('oof f1', metrics.f1_score(oof, y))
result = test[["manufacturer", "model", "serial_number", "dt"]]
result['pred0'] = pred
pred = [1 if p > 0.5 else 0 for p in pred]
result['pred'] = pred
print(result.shape)  # (144915, 5)
result0 = result
result = result[result.pred == 1]
print(result.shape)  # （135419, 5）
result[["manufacturer", "model", "serial_number", "dt"]].to_csv(path + "/result/result_month56_0.25.csv", index=False, header=None)
result = result.groupby(["manufacturer", "model", "serial_number"])["dt"].min().reset_index()
result.to_csv(path + '/result/result_month56_0.25_min.csv', index=False, header=None)
print(result.shape)  # （9077， 4）
#


# In[85]:


import plotly.express as px
data = result0[["manufacturer", "model", "serial_number",'pred0','pred','dt']]
dt = data.sort_values(by=["manufacturer", "model",'serial_number','pred0'], ascending=False)
print(dt.shape)
dt.head()


# In[86]:


dt2 = dt.drop_duplicates(subset=["manufacturer", "model", "serial_number"], keep='first')
print(dt2.shape)
dt2.head()
#dt = data.groupby(['pred0']).count
#tf2 = tf1.sort_values(by=['serial_number','dt'], ascending=False)
#tfd2 = tf2.drop_duplicates(subset=["manufacturer", "model", "serial_number"], keep='first')


# In[87]:


dt21 = dt2[dt2['pred0']==1]
print(dt21.shape)
dt21.head()


# In[ ]:


tg = result.groupby(["manufacturer", "model", "serial_number"])['dt'].max().reset_index()


# In[ ]:


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
tr.to_csv(path + '/result/result_month56_max_test.csv', index=False, header=None)
print(tr.shape)
tr.head()


# In[ ]:


# 把异常数据结果和正常数据结果拼接起来
ret1 = pd.read_csv(path + '/result/result_abnormal.csv',header=["manufacturer", "model", "serial_number",'dt'])
ret0 = pd.read_csv(path + '/result/result_month56_max_test.csv',header=["manufacturer", "model", "serial_number",'dt'])
ret = pd.concat([ret1, ret0])
ret.to_csv(path + '/result/result_month56.csv', index=False,header=False)
print(ret.shape)

