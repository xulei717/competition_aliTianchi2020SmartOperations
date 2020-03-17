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


# 第一类正负样本，测试集：(7505, 46)，(60000, 46)，（33111, 43）
# 第二类正负样本，测试集：(16919, 46)，(60000, 46)，（177946, 43）
# 第一类和第二类预测结果，总结果数：(46, 4) (536, 4) (541, 4)


# In[4]:


# 获取第一类正负样本，并预测第一类测试集
tt = pd.read_csv(path + '/data/processed_date_data/traintd1.csv')
print(tt.shape)  # (7505, 46)
print(tt.columns)
tf = pd.read_csv(path + '/data/processed_date_data/trainfd1.csv')
tf = tf.sample(frac=0.5, random_state=0)
print(tf.shape)  # (60000, 46)
print(tf.columns)
test = pd.read_csv(path + '/data/processed_date_data/testd1.csv')
print(test.shape)  # （33111, 43）
print(test.columns)
# 把正负样本合并一起
train = pd.concat([tt, tf])
print(train.shape)  # (67505, 46)


# In[5]:


dels = ['smart_184raw', 'smart_184_normalized', 'smart_4_normalized']
features = [x for x in train.columns if x not in dels + ["serial_number", "manufacturer", "model", "dt", "fault_time", "differ_day", "label", 'dt_first']]
print(train[features].shape, test[features].shape)
print(set(train[features].columns)-set(test[features].columns))
target = 'label'


# In[6]:


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
    joblib.dump(model, path + '/models/model1_dels_sm0.5_' + str(val_f1) + '.txt')

    test_pred = model.predict(test[features])
    pred += test_pred / 5

print('oof f1', metrics.f1_score(oof, y))
result = test[["manufacturer", "model", "serial_number", "dt"]]
pred = [1 if p > 0.5 else 0 for p in pred]
result['pred'] = pred
print(result.shape)  # (33111, 5)
result = result[result.pred == 1]
print(result.shape)  # (370, 5)
result[["manufacturer", "model", "serial_number", "dt"]].to_csv(path + "/result/result1_dels_sm0.5.csv", index=False, header=None)
result = result.groupby(["manufacturer", "model", "serial_number"])["dt"].min().reset_index()
result.to_csv(path + '/result/result1_dels_sm0.5_min.csv', index=False, header=None)
print(result.shape)  # (46, 4)


# In[7]:


# 获取第二类正负样本，并预测第二类测试集
tt = pd.read_csv(path + '/data/processed_date_data/traintd2.csv')
print(tt.shape)  # (16919, 46)
print(tt.columns)
tf = pd.read_csv(path + '/data/processed_date_data/trainfd2.csv')
tf = tf.sample(frac=0.5, random_state=0)
print(tf.shape)  # (60000, 46)
print(tf.columns)
test = pd.read_csv(path + '/data/processed_date_data/testd2.csv')
print(test.shape)  # （177946, 43）
print(test.columns)
# 把正负样本合并一起
train = pd.concat([tt, tf])
print(train.shape)  # （76919， 46）


# In[8]:


dels = ['smart_4_normalized','smart_192_normalized','smart_12_normalized',
       'smart_197_normalized','smart_188_normalized','smart_198_normalized']
features = [x for x in train.columns if x not in dels+["serial_number", "manufacturer", "model", "dt", "fault_time", "differ_day", "label", 'dt_first']]
print(train[features].shape, test[features].shape)
print(set(train[features].columns)-set(test[features].columns))
target = 'label'


# In[9]:


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
    joblib.dump(model, path + '/models/model2_dels_sm0.5_' + str(val_f1) + '.txt')

    test_pred = model.predict(test[features])
    pred += test_pred / 5

print('oof f1', metrics.f1_score(oof, y))
result = test[["manufacturer", "model", "serial_number", "dt"]]
pred = [1 if p > 0.5 else 0 for p in pred]
result['pred'] = pred
print(result.shape)  # (177946, 5)
result = result[result.pred == 1]
print(result.shape)  # (7066, 5)
result[["manufacturer", "model", "serial_number", "dt"]].to_csv(path + "/result/result2_dels_sm0.5.csv", index=False, header=None)
result = result.groupby(["manufacturer", "model", "serial_number"])["dt"].min().reset_index()
result.to_csv(path + '/result/result2_dels_sm0.5_min.csv', index=False, header=None)
print(result.shape)  # (536, 4)


# In[10]:


# 把一、二类预测结果进行合并
pre1 = pd.read_csv(path + '/result/result1_dels_sm0.5_min.csv', names=["manufacturer", "model", "serial_number", "dt"])
pre2 = pd.read_csv(path + '/result/result2_dels_sm0.5_min.csv', names=["manufacturer", "model", "serial_number", "dt"])
pre = pd.concat([pre1,pre2])
pre.drop_duplicates(inplace=True)
pre = pre.groupby(["manufacturer", "model", "serial_number"])['dt'].min().reset_index()
pre.to_csv(path + '/result/result12_dels_sm0.5_min.csv', index=False, header=None)
print(pre1.shape, pre2.shape, pre.shape)  # (46, 4) (536, 4) (541, 4)


# In[11]:


# 把最终结果result12_dels_min.csv里的dt转换成test里最晚出现的时间
test = pd.read_csv(path + '/data/processed_date_data/test.csv')
td = test[test['model']==1]
td = td[['serial_number','dt']].groupby(['serial_number'])['dt'].max().reset_index()
print(td.shape)
rt12 = pd.read_csv(path + '/result/result12_dels_sm0.5_min.csv', names=["manufacturer", "model", "serial_number", "dt"])
tr = rt12.merge(td, how='left', on='serial_number')
print(tr.shape)
tr['dt'] = tr['dt_y']
del tr['dt_x']
del tr['dt_y']
tr.drop_duplicates(inplace=True)
tr.reset_index(drop=True, inplace=True)
tr.to_csv(path + '/result/result12_dels_sm0.5_max_test.csv', index=False, header=None)
print(tr.shape)
tr.head()


# In[12]:


# 把最终结果result12_min.csv里的dt转换成test里最早出现的时间
test = pd.read_csv(path + '/data/processed_date_data/test.csv')
td = test[test['model']==1]
td = td[['serial_number','dt']].groupby(['serial_number'])['dt'].min().reset_index()
print(td.shape)
rt12 = pd.read_csv(path + '/result/result12_dels_sm0.5_min.csv', names=["manufacturer", "model", "serial_number", "dt"])
tr = rt12.merge(td, how='left', on='serial_number')
print(tr.shape)
tr['dt'] = tr['dt_y']
del tr['dt_x']
del tr['dt_y']
tr.drop_duplicates(inplace=True)
tr.reset_index(drop=True, inplace=True)
tr.to_csv(path + '/result/result12_dels_sm0.5_min_test.csv', index=False, header=None)
print(tr.shape)
tr.head()


# In[ ]:




