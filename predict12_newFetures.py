#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import os
path = os.getcwd()
path


# In[ ]:


# 第一类正负样本，测试集：(7505, 46)，(120000, 46)，（33111, 43）
# 第二类正负样本，测试集：(16919, 46)，(120000, 46)，（177946, 43）
# 第一类和第二类预测结果，总结果数：(19, 4) (375, 4) (378, 4)


# In[12]:


# 获取第一类正负样本，并预测第一类测试集
tt = pd.read_csv(path + '/data/processed_date_data/traintd1.csv')
print(tt.shape)  # (7505, 46)
tf = pd.read_csv(path + '/data/processed_date_data/trainfd1.csv')
print(tf.shape)  # (120000, 46)
test = pd.read_csv(path + '/data/processed_date_data/testd1.csv')
print(test.shape)  # （33111, 43）
# 把正负样本合并一起
train = pd.concat([tt, tf])
print(train.shape)


# In[14]:


#features = [x for x in train.columns if x not in ["serial_number", "manufacturer", "model", "dt", "fault_time", "differ_day", "label", 'dt_first']]
features = ['smart_1_normalized','smart_3_normalized','smart_5raw','smart_184raw','smart_187raw',
           'smart_188raw','smart_189raw','smart_192raw','smart_193raw','smart_194raw','smart_197raw',
           'smart_198raw','smart_199raw','days']
print(train[features].shape, test[features].shape)
print(set(train[features].columns)-set(test[features].columns))
target = 'label'


# In[13]:


train['smart_5raw'] = train['smart_5raw'].apply(lambda x: 1 if x > 0 else 0)
train['smart_187raw'] = train['smart_187raw'].apply(lambda x: 1 if x > 0 else 0)
train['smart_188raw'] = train['smart_188raw'].apply(lambda x: 1 if x > 0 else 0)
train['smart_197raw'] = train['smart_197raw'].apply(lambda x: 1 if x > 0 else 0)
train['smart_198raw'] = train['smart_198raw'].apply(lambda x: 1 if x > 0 else 0)
train[['smart_5raw','smart_187raw','smart_188raw','smart_197raw','smart_198raw']].describe()


# In[15]:


test['smart_5raw'] = test['smart_5raw'].apply(lambda x: 1 if x > 0 else 0)
test['smart_187raw'] = test['smart_187raw'].apply(lambda x: 1 if x > 0 else 0)
test['smart_188raw'] = test['smart_188raw'].apply(lambda x: 1 if x > 0 else 0)
test['smart_197raw'] = test['smart_197raw'].apply(lambda x: 1 if x > 0 else 0)
test['smart_198raw'] = test['smart_198raw'].apply(lambda x: 1 if x > 0 else 0)
test[['smart_5raw','smart_187raw','smart_188raw','smart_197raw','smart_198raw']].describe()


# In[16]:


# lightgbm模型
fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)

X = train[features].copy()
y = train[target]
#models = list()
pred = numpy.zeros(len(test))
oof = numpy.zeros(len(X))

for index, (train_idx, val_idx) in enumerate(fold.split(X, y)):
    train_X = X.iloc[train_idx]
    train_y = y.iloc[train_idx]
    val_X = X.iloc[val_idx]
    val_y = y.iloc[val_idx]
    model = lightgbm.LGBMClassifier(
        boosting_type="gbdt",
        metric='auc',
        learning_rate=0.001,
        n_estimators=5000,
        num_leaves=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=2020)
    model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=500)
    #models.append(model)
    val_pred = model.predict(val_X)
    oof[val_idx] = val_pred
    val_f1 = metrics.f1_score(val_y, val_pred)
    print(index, 'val f1', val_f1)
    joblib.dump(model, path + '/models/model1_auc_newfeatures_' + str(val_f1) + '.txt')

    test_pred = model.predict(test[features])
    pred += test_pred / 5

print('oof f1', metrics.f1_score(oof, y))
result = test[["manufacturer", "model", "serial_number", "dt"]]
pred = [1 if p > 0.5 else 0 for p in pred]
result['pred'] = pred
print(result.shape)  # (33111, 5)
result = result[result.pred == 1]
print(result.shape)  # (329, 5)
result[["manufacturer", "model", "serial_number", "dt"]].to_csv(path + "/result/result1_auc_newfeatures.csv", index=False, header=None)
result = result.groupby(["manufacturer", "model", "serial_number"])["dt"].min().reset_index()
result.to_csv(path + '/result/result1_auc_newfeatures_min.csv', index=False, header=None)
print(result.shape)  # (41, 4)


# In[17]:


# 获取第二类正负样本，并预测第二类测试集
tt = pd.read_csv(path + '/data/processed_date_data/traintd2.csv')
print(tt.shape)  # (16919, 46)
print(tt.columns)
tf = pd.read_csv(path + '/data/processed_date_data/trainfd2.csv')
print(tf.shape)  # (120000, 46)
print(tf.columns)
test = pd.read_csv(path + '/data/processed_date_data/testd2.csv')
print(test.shape)  # （177946, 43）
print(test.columns)
# 把正负样本合并一起
train = pd.concat([tt, tf])
print(train.shape)


# In[19]:


train['smart_5raw'] = train['smart_5raw'].apply(lambda x: 1 if x > 0 else 0)
train['smart_187raw'] = train['smart_187raw'].apply(lambda x: 1 if x > 0 else 0)
train['smart_188raw'] = train['smart_188raw'].apply(lambda x: 1 if x > 0 else 0)
train['smart_197raw'] = train['smart_197raw'].apply(lambda x: 1 if x > 0 else 0)
train['smart_198raw'] = train['smart_198raw'].apply(lambda x: 1 if x > 0 else 0)
train[['smart_5raw','smart_187raw','smart_188raw','smart_197raw','smart_198raw']].describe()


# In[20]:


test['smart_5raw'] = test['smart_5raw'].apply(lambda x: 1 if x > 0 else 0)
test['smart_187raw'] = test['smart_187raw'].apply(lambda x: 1 if x > 0 else 0)
test['smart_188raw'] = test['smart_188raw'].apply(lambda x: 1 if x > 0 else 0)
test['smart_197raw'] = test['smart_197raw'].apply(lambda x: 1 if x > 0 else 0)
test['smart_198raw'] = test['smart_198raw'].apply(lambda x: 1 if x > 0 else 0)
test[['smart_5raw','smart_187raw','smart_188raw','smart_197raw','smart_198raw']].describe()


# In[21]:


#features = [x for x in train.columns if x not in ["serial_number", "manufacturer", "model", "dt", "fault_time", "differ_day", "label", 'dt_first']]
features = ['smart_1_normalized','smart_3_normalized','smart_5raw','smart_184raw','smart_187raw',
           'smart_188raw','smart_189raw','smart_192raw','smart_193raw','smart_194raw','smart_197raw',
           'smart_198raw','smart_199raw','days']
print(train[features].shape, test[features].shape)
target = 'label'


# In[22]:


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
        metric='auc',
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
    joblib.dump(model, path + '/models/model2_auc_newfeatures_' + str(val_f1) + '.txt')


    test_pred = model.predict(test[features])
    pred += test_pred / 5

print('oof f1', metrics.f1_score(oof, y))
result = test[["manufacturer", "model", "serial_number", "dt"]]
pred = [1 if p > 0.5 else 0 for p in pred]
result['pred'] = pred
print(result.shape)  # (177946, 5)
result = result[result.pred == 1]
print(result.shape)  # (7498, 5)   
result[["manufacturer", "model", "serial_number", "dt"]].to_csv(path + "/result/result2_auc_newfeatures.csv", index=False, header=None)
result = result.groupby(["manufacturer", "model", "serial_number"])["dt"].min().reset_index()
result.to_csv(path + '/result/result2_auc_newfeatures_min.csv', index=False, header=None)
print(result.shape)  # (601, 4)


# In[23]:


# 把一、二类预测结果进行合并
pre1 = pd.read_csv(path + '/result/result1_auc_newfeatures_min.csv', names=["manufacturer", "model", "serial_number", "dt"])
pre2 = pd.read_csv(path + '/result/result2_auc_newfeatures_min.csv', names=["manufacturer", "model", "serial_number", "dt"])
pre = pd.concat([pre1,pre2])
pre.drop_duplicates(inplace=True)
pre = pre.groupby(["manufacturer", "model", "serial_number"])['dt'].min().reset_index()
pre.to_csv(path + '/result/result12_auc_newfeatures_min.csv', index=False, header=None)
print(pre1.shape, pre2.shape, pre.shape)  # (41, 4) (601, 4) (602, 4)


# # 把最终结果result12_min.csv里的dt转换成test里最晚出现的时间

# In[24]:


# 把最终结果result12_min.csv里的dt转换成test里最晚出现的时间
test = pd.read_csv(path + '/data/processed_date_data/test.csv')
td = test[test['model']==1]
td = td[['serial_number','dt']].groupby(['serial_number'])['dt'].max().reset_index()
print(td.shape)
td.head()


# In[25]:


rt12 = pd.read_csv(path + '/result/result12_auc_newfeatures_min.csv', names=["manufacturer", "model", "serial_number", "dt"])
tr = rt12.merge(td, how='left', on='serial_number')
print(tr.shape)
tr.head()


# In[26]:


tr['dt'] = tr['dt_y']
del tr['dt_x']
del tr['dt_y']
tr.head()


# In[27]:


tr.drop_duplicates(inplace=True)
tr.reset_index(drop=True, inplace=True)
print(tr.shape)
tr.head()


# In[28]:


tr.to_csv(path + '/result/result12_auc_newfeatures_max_test.csv', index=False, header=None)


# In[ ]:




