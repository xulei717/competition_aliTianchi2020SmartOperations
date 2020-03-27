#!/usr/bin/env python
# coding: utf-8

# In[7]:


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


# In[8]:


# 正样本8类大小：(3086, 52) (3655, 52) (2816, 52) (1455, 52),(8105, 52) (4662, 52) (4324, 52) (1885, 52)
# 测试集8类大小：(89, 49) (3416, 49) (15442, 49) (14164, 49),(960, 49) (23532, 49) (58286, 49) (62137, 49)
# 负样本8类大小：(28403, 44) (36540, 44) (28152, 44) (14544, 44)，(81048, 44) (46620, 44) (43236, 44) (18840, 44)


# In[10]:


#features = [x for x in train.columns if x not in ["serial_number", "manufacturer", "model", "dt", "fault_time", "differ_day", "label", 'dt_first']]
features = ['smart_1_normalized','smart_3_normalized','smart_5raw','smart_184raw','smart_187raw',
           'smart_188raw','smart_189raw','smart_192raw','smart_193raw','smart_194raw','smart_197raw',
           'smart_198raw','smart_199raw','days']
print(train[features].shape, test[features].shape)
print(set(train[features].columns)-set(test[features].columns))
target = 'label'


# In[38]:


# 获取正负样本，并预测测试集
cla = '04'
tt = pd.read_csv(path + '/data/processed_date_data/traintd'+cla+'.csv')
print(tt.shape)  # (7505, 46)
tf = pd.read_csv(path + '/data/processed_date_data/trainfd'+cla+'.csv')
print(tf.shape)  # (120000, 46)
test = pd.read_csv(path + '/data/processed_date_data/testd'+cla+'.csv')
print(test.shape)  # （33111, 43）
# 把正负样本合并一起
train = pd.concat([tt, tf])
print(train.shape)


# In[39]:


train['smart_5raw'] = train['smart_5raw'].apply(lambda x: 1 if x > 0 else 0)
train['smart_187raw'] = train['smart_187raw'].apply(lambda x: 1 if x > 0 else 0)
train['smart_188raw'] = train['smart_188raw'].apply(lambda x: 1 if x > 0 else 0)
train['smart_197raw'] = train['smart_197raw'].apply(lambda x: 1 if x > 0 else 0)
train['smart_198raw'] = train['smart_198raw'].apply(lambda x: 1 if x > 0 else 0)
train[['smart_5raw','smart_187raw','smart_188raw','smart_197raw','smart_198raw']].describe()


# In[40]:


test['smart_5raw'] = test['smart_5raw'].apply(lambda x: 1 if x > 0 else 0)
test['smart_187raw'] = test['smart_187raw'].apply(lambda x: 1 if x > 0 else 0)
test['smart_188raw'] = test['smart_188raw'].apply(lambda x: 1 if x > 0 else 0)
test['smart_197raw'] = test['smart_197raw'].apply(lambda x: 1 if x > 0 else 0)
test['smart_198raw'] = test['smart_198raw'].apply(lambda x: 1 if x > 0 else 0)
test[['smart_5raw','smart_187raw','smart_188raw','smart_197raw','smart_198raw']].describe()


# In[41]:


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
    joblib.dump(model, path + '/models/model'+cla+'_auc_newfeatures_' + str(val_f1) + '.txt')

    test_pred = model.predict(test[features])
    pred += test_pred / 5

print('oof f1', metrics.f1_score(oof, y))
result = test[["manufacturer", "model", "serial_number", "dt"]]
pred = [1 if p > 0.5 else 0 for p in pred]
result['pred'] = pred
print(result.shape) 
result = result[result.pred == 1]
print(result.shape)  
result[["manufacturer", "model", "serial_number", "dt"]].to_csv(path + "/result/result"+cla+"_auc_newfeatures8.csv", index=False, header=None)
result = result.groupby(["manufacturer", "model", "serial_number"])["dt"].min().reset_index()
result.to_csv(path + '/result/result'+cla+'_auc_newfeatures_min8.csv', index=False, header=None)
print(result.shape) 
# 11: (89, 5), (0, 5), (0, 2)
# 12: (3416, 5), (49, 5), (10, 4)
# 13: (15442, 5), (81, 5), (20, 4)
# 14: (14164, 5), (195, 5), (20, 4)
# 01: (960, 5), (0, 5), (0, 2)
# 02: (23532, 5), (83, 5), (15, 2)
# 03: (58286, 5), (55, 5), (16, 4)
# 04: (62137, 5), (1337, 5), (185, 4)


# In[42]:


# 把8类预测结果进行合并
pre11 = pd.read_csv(path + '/result/result11_auc_newfeatures_min8.csv', names=["manufacturer", "model", "serial_number", "dt"])
pre12 = pd.read_csv(path + '/result/result12_auc_newfeatures_min8.csv', names=["manufacturer", "model", "serial_number", "dt"])
pre13 = pd.read_csv(path + '/result/result13_auc_newfeatures_min8.csv', names=["manufacturer", "model", "serial_number", "dt"])
pre14 = pd.read_csv(path + '/result/result14_auc_newfeatures_min8.csv', names=["manufacturer", "model", "serial_number", "dt"])
pre01 = pd.read_csv(path + '/result/result01_auc_newfeatures_min8.csv', names=["manufacturer", "model", "serial_number", "dt"])
pre02 = pd.read_csv(path + '/result/result02_auc_newfeatures_min8.csv', names=["manufacturer", "model", "serial_number", "dt"])
pre03 = pd.read_csv(path + '/result/result03_auc_newfeatures_min8.csv', names=["manufacturer", "model", "serial_number", "dt"])
pre04 = pd.read_csv(path + '/result/result04_auc_newfeatures_min8.csv', names=["manufacturer", "model", "serial_number", "dt"])
pre = pd.concat([pre11,pre12,pre13,pre14,pre01,pre02,pre03,pre04])
pre.drop_duplicates(inplace=True)
pre = pre.groupby(["manufacturer", "model", "serial_number"])['dt'].min().reset_index()
pre.to_csv(path + '/result/result8_auc_newfeatures_min.csv', index=False, header=None)
print(pre11.shape, pre12.shape, pre13.shape, pre14.shape)
print(pre01.shape, pre02.shape, pre03.shape, pre04.shape)
print(pre.shape)


# # 把最终结果里的dt转换成test里最晚出现的时间

# In[43]:


# 把最终结果result12_min.csv里的dt转换成test里最晚出现的时间
test = pd.read_csv(path + '/data/processed_date_data/test.csv')
td = test[test['model']==1]
td = td[['serial_number','dt']].groupby(['serial_number'])['dt'].max().reset_index()
print(td.shape)
td.head()


# In[44]:


rt12 = pd.read_csv(path + '/result/result8_auc_newfeatures_min.csv', names=["manufacturer", "model", "serial_number", "dt"])
tr = rt12.merge(td, how='left', on='serial_number')
print(tr.shape)
tr.head()


# In[45]:


tr['dt'] = tr['dt_y']
del tr['dt_x']
del tr['dt_y']
tr.head()


# In[46]:


tr.drop_duplicates(inplace=True)
tr.reset_index(drop=True, inplace=True)
print(tr.shape)
tr.head()


# In[47]:


tr.to_csv(path + '/result/result8_auc_newfeatures_max_test.csv', index=False, header=None)


# In[ ]:




