#!/usr/bin/env python
# coding: utf-8

# In[89]:


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


# In[90]:


# 获取第一类正负样本，并预测第一类测试集
tt = pd.read_csv(path + '/data/processed_date_data/traintd1.csv')
print(tt.shape)  # (7505, 46)
print(tt.columns)
tf = pd.read_csv(path + '/data/processed_date_data/trainfd1.csv')
print(tf.shape)  # (120000, 46)
print(tf.columns)
test = pd.read_csv(path + '/data/processed_date_data/testd1.csv')
print(test.shape)  # （33111, 43）
print(test.columns)


# In[91]:


# 把正负样本合并一起
train = pd.concat([tt, tf])
print(train.shape)


# In[92]:


features = [x for x in train.columns if x not in ["serial_number", "manufacturer", "model", "dt", "fault_time", "differ_day", "label", 'dt_first']]
print(train[features].shape, test[features].shape)
print(set(train[features].columns)-set(test[features].columns))
target = 'label'


# In[94]:


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
    joblib.dump(model, path + '/models/model1_auc_' + str(val_f1) + '.txt')

    test_pred = model.predict(test[features])
    pred += test_pred / 5

print('oof f1', metrics.f1_score(oof, y))
result = test[["manufacturer", "model", "serial_number", "dt"]]
pred = [1 if p > 0.5 else 0 for p in pred]
result['pred'] = pred
print(result.shape)  # (33111, 5)
result = result[result.pred == 1]
print(result.shape)  # (107, 5)
result[["manufacturer", "model", "serial_number", "dt"]].to_csv(path + "/result/result1_auc.csv", index=False, header=None)
result = result.groupby(["manufacturer", "model", "serial_number"])["dt"].min().reset_index()
result.to_csv(path + '/result/result1_auc_min.csv', index=False, header=None)
print(result.shape)  # (19, 4)


# In[95]:


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


# In[96]:


features = [x for x in train.columns if x not in ["serial_number", "manufacturer", "model", "dt", "fault_time", "differ_day", "label", 'dt_first']]
print(train[features].shape, test[features].shape)
print(set(train[features].columns)-set(test[features].columns))
target = 'label'


# In[97]:


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
    joblib.dump(model, path + '/models/model2_auc_' + str(val_f1) + '.txt')


    test_pred = model.predict(test[features])
    pred += test_pred / 5

print('oof f1', metrics.f1_score(oof, y))
result = test[["manufacturer", "model", "serial_number", "dt"]]
pred = [1 if p > 0.5 else 0 for p in pred]
result['pred'] = pred
print(result.shape)  # (177946, 5)
result = result[result.pred == 1]
print(result.shape)  # (4802, 5)   
result[["manufacturer", "model", "serial_number", "dt"]].to_csv(path + "/result/result2_auc.csv", index=False, header=None)
result = result.groupby(["manufacturer", "model", "serial_number"])["dt"].min().reset_index()
result.to_csv(path + '/result/result2_auc_min.csv', index=False, header=None)
print(result.shape)  # (375, 4)


# In[98]:


# 把一、二类预测结果进行合并
pre1 = pd.read_csv(path + '/result/result1_auc_min.csv', names=["manufacturer", "model", "serial_number", "dt"])
pre2 = pd.read_csv(path + '/result/result2_auc_min.csv', names=["manufacturer", "model", "serial_number", "dt"])
pre = pd.concat([pre1,pre2])
pre.drop_duplicates(inplace=True)
pre = pre.groupby(["manufacturer", "model", "serial_number"])['dt'].min().reset_index()
pre.to_csv(path + '/result/result12_auc_min.csv', index=False, header=None)
print(pre1.shape, pre2.shape, pre.shape)  # (19, 4) (375, 4) (378, 4)


# In[34]:


# 分析最终结果的dt列时间分布
pre = pd.read_csv(path + '/result/result12_auc_min.csv', names=["manufacturer", "model", "serial_number", "dt"])
print(pre.shape)
tm = pre['dt'].value_counts().reset_index()
tm.columns = ['dt','count']
tm = tm.sort_values(by=['dt'],axis=0,ascending=True)
print(tm.shape)
tm.head()


# In[41]:


# 画柱状图
import random
import plotly as py
import plotly.graph_objs as go
#import plotly.graph_objs.layout.Legend as lgd
pyplt = py.offline.plot

trace0 = go.Bar(
    x = tm['dt'],
    y = tm['count'],
    name = 'count',
    text = [str(i) for i in tm['count']],
    textposition = "outside",
    insidetextanchor = 'middle',
    hovertemplate = '[x:%{x},y:%{y}]',
)

trace = [trace0]
layout = go.Layout(
    title = dict(text='dt_count',x=0.5,y=0.93),
    showlegend = True,
    legend = dict(x=1, y=0.5),
)
fig = go.Figure(data=trace, layout=layout)
pyplt(fig, filename=path + '/result/dt_count.html')


# In[44]:


import plotly.express as px
fig = px.bar(tm, x='dt', y='count')
fig.show()


# # 把最终结果result12_min.csv里的dt转换成test里最晚出现的时间

# In[99]:


# 把最终结果result12_min.csv里的dt转换成test里最晚出现的时间
test = pd.read_csv(path + '/data/processed_date_data/test.csv')
td = test[test['model']==1]
td = td[['serial_number','dt']].groupby(['serial_number'])['dt'].max().reset_index()
print(td.shape)
td.head()


# In[100]:


rt12 = pd.read_csv(path + '/result/result12_auc_min.csv', names=["manufacturer", "model", "serial_number", "dt"])
tr = rt12.merge(td, how='left', on='serial_number')
print(tr.shape)
tr.head()


# In[101]:


tr['dt'] = tr['dt_y']
del tr['dt_x']
del tr['dt_y']
tr.head()


# In[102]:


tr.drop_duplicates(inplace=True)
tr.reset_index(drop=True, inplace=True)
print(tr.shape)
tr.head()


# In[103]:


tr.to_csv(path + '/result/result12_auc_max_test.csv', index=False, header=None)


# In[42]:


# 把最终结果result12_min.csv里的dt转换成test里最早出现的时间
test = pd.read_csv(path + '/data/processed_date_data/test.csv')
td = test[test['model']==1]
td = td[['serial_number','dt']].groupby(['serial_number'])['dt'].min().reset_index()
print(td.shape)
rt12 = pd.read_csv(path + '/result/result12_min.csv', names=["manufacturer", "model", "serial_number", "dt"])
tr = rt12.merge(td, how='left', on='serial_number')
print(tr.shape)
tr['dt'] = tr['dt_y']
del tr['dt_x']
del tr['dt_y']
tr.drop_duplicates(inplace=True)
tr.reset_index(drop=True, inplace=True)
tr.to_csv(path + '/result/result12_min_test.csv', index=False, header=None)
print(tr.shape)
tr.head()


# # 查看model的重要特征分布

# In[58]:


models = []
for m in os.listdir(path + '/models/'):
    models.append(m)
print(len(models))
models[0]


# In[74]:


md = joblib.load(path + '/models/' + models[0])
print(md.feature_importances_)
print(md.n_features_, md.best_score_, md.best_iteration_)
print(md.booster_.feature_name())


# In[76]:


# 画柱状图
import random
import plotly as py
import plotly.graph_objs as go
#import plotly.graph_objs.layout.Legend as lgd
pyplt = py.offline.plot

def feature_Bar(model_name):
    md = joblib.load(path + '/models/' + model_name)
    trace0 = go.Bar(
        x = md.booster_.feature_name(),
        y = md.feature_importances_,
        name = 'feature_importance',
        text = [str(i) for i in md.feature_importances_],
        textposition = "outside",
        insidetextanchor = 'middle',
        hovertemplate = '[x:%{x},y:%{y}]',
    )

    trace = [trace0]
    layout = go.Layout(
        title = dict(text=model_name,x=0.5,y=0.93),
        showlegend = True,
        legend = dict(x=1, y=0.5),
    )
    fig = go.Figure(data=trace, layout=layout)
    pyplt(fig, filename=path + '/feature_importance/'+model_name+'.html')


# In[77]:


for mm in models:
    feature_Bar(mm)


# In[78]:


lightgbm.plot_importance(md, max_num_features=38)


# In[80]:


# 统计每个特征的十个模型的重要值
mf = {}
md0 = joblib.load(path + '/models/' + models[0])
x = md.booster_.feature_name()
y = md.feature_importances_
for i in range(len(x)):
    mf[x[i]] = [y[i]]
for m in models:
    md = joblib.load(path + '/models/' + m)
    x = md.booster_.feature_name()
    y = md.feature_importances_
    for i in range(len(x)):
        mf[x[i]].append(y[i])
print(mf)


# In[83]:


mfr = {}
for k,v in mf.items():
    mfr[k] = [min(v), max(v), sum(v)//len(v)]
mfr


# In[84]:


# 按照最后一个平均值排序
mfrs = sorted(mfr.items(), key=lambda x:x[1][2], reverse=False)
mfrs


# In[85]:


models


# In[87]:


# 统计第一类每个特征的5个模型的重要值
mf = {}
md0 = joblib.load(path + '/models/' + models[1])
x = md.booster_.feature_name()
y = md.feature_importances_
for i in range(len(x)):
    mf[x[i]] = [y[i]]
for m in models[2:6]:
    md = joblib.load(path + '/models/' + m)
    x = md.booster_.feature_name()
    y = md.feature_importances_
    for i in range(len(x)):
        mf[x[i]].append(y[i])
#print(mf)
mfr = {}
for k,v in mf.items():
    mfr[k] = [min(v), max(v), sum(v)//len(v)]
#print(mfr)
# 按照最后一个平均值排序
mfrs = sorted(mfr.items(), key=lambda x:x[1][2], reverse=False)
print(mfrs)


# In[88]:


# 统计第二类每个特征的5个模型的重要值
mf = {}
md0 = joblib.load(path + '/models/' + models[0])
x = md.booster_.feature_name()
y = md.feature_importances_
for i in range(len(x)):
    mf[x[i]] = [y[i]]
for m in models[6:]:
    md = joblib.load(path + '/models/' + m)
    x = md.booster_.feature_name()
    y = md.feature_importances_
    for i in range(len(x)):
        mf[x[i]].append(y[i])
#print(mf)
mfr = {}
for k,v in mf.items():
    mfr[k] = [min(v), max(v), sum(v)//len(v)]
#print(mfr)
# 按照最后一个平均值排序
mfrs = sorted(mfr.items(), key=lambda x:x[1][2], reverse=False)
print(mfrs)


# 
