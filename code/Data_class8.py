#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
import pandas as pd
import numpy as np
from tqdm import tqdm
path = os.getcwd()
path


# # 1. 把训练集中的正样本按5|187|188|197|198＞0分成两份 然后按smart_9raw分成4份 以≤1万 1-2万 2-3万 ＞3万为区间，分别统计各条件下的正样本数（共8份）
# 
# # 2. 在训练集中，以各条件下的正样本数×10的数量对负样本采样 同时每个条件下的负样本按月均匀分布 如果前面的月份数量不够可用后面的月份补
# 
# # 3. 用每个条件下的正负样本训练模型 去预测对应条件下的测试样本的label 后续生成结果的方案不变

# In[ ]:


# 正样本8类大小：(3086, 52) (3655, 52) (2816, 52) (1455, 52),(8105, 52) (4662, 52) (4324, 52) (1885, 52)
# 测试集8类大小：(89, 49) (3416, 49) (15442, 49) (14164, 49),(960, 49) (23532, 49) (58286, 49) (62137, 49)
# 负样本8类大小：(28403, 44) (36540, 44) (28152, 44) (14544, 44)，(81048, 44) (46620, 44) (43236, 44) (18840, 44)


# In[4]:


# 先把正样本分成8份
traint = pd.read_csv(path + '/data/processed_date_data/traint.csv')
print(traint.shape)  # (29992, 52)
traint.head(2)


# In[14]:


traint1 = traint[(traint['smart_5raw'] > 0) | 
                (traint['smart_187raw'] > 0) | 
                (traint['smart_188raw'] > 0) | 
                (traint['smart_197raw'] > 0) | 
                (traint['smart_198raw'] > 0)]  # 筛选raw:5,187,188,197,198任一大于0
print(traint1.shape)  # (11012, 52)
traint11 = traint1[traint1['smart_9raw'] <= 10000]
traint12 = traint1[(traint1['smart_9raw'] > 10000) & (traint1['smart_9raw'] <= 20000)]
traint13 = traint1[(traint1['smart_9raw'] > 20000) & (traint1['smart_9raw'] <= 30000)]
traint14 = traint1[(traint1['smart_9raw'] > 30000)]
print(traint11.shape, traint12.shape, traint13.shape, traint14.shape)


# In[6]:


traint0 = traint[(traint['smart_5raw']==0) & 
                (traint['smart_187raw']==0) & 
                (traint['smart_188raw']==0) & 
                (traint['smart_197raw']==0) & 
                (traint['smart_198raw']==0)]  # 筛选raw:5,187,188,197,198全为0
print(traint0.shape)  # (18976, 52)


# In[15]:


traint01 = traint0[traint0['smart_9raw'] <= 10000]
traint02 = traint0[(traint0['smart_9raw'] > 10000) & (traint0['smart_9raw'] <= 20000)]
traint03= traint0[(traint0['smart_9raw'] > 20000) & (traint0['smart_9raw'] <= 30000)]
traint04 = traint0[(traint0['smart_9raw'] > 30000)]
print(traint01.shape, traint02.shape, traint03.shape, traint04.shape)
# 


# In[17]:


traint11.to_csv(path + '/data/processed_date_data/traint11.csv', index=False)
traint12.to_csv(path + '/data/processed_date_data/traint12.csv', index=False)
traint13.to_csv(path + '/data/processed_date_data/traint13.csv', index=False)
traint14.to_csv(path + '/data/processed_date_data/traint14.csv', index=False)
traint01.to_csv(path + '/data/processed_date_data/traint01.csv', index=False)
traint02.to_csv(path + '/data/processed_date_data/traint02.csv', index=False)
traint03.to_csv(path + '/data/processed_date_data/traint03.csv', index=False)
traint04.to_csv(path + '/data/processed_date_data/traint04.csv', index=False)


# In[20]:


test = pd.read_csv(path + '/data/processed_date_data/test.csv')
print(test.shape)  # (178028, 49)
test1 = test[(test['smart_198raw'] > 0)] #& 
#                 (test['smart_187raw'] > 0) & 
#                 (test['smart_188raw'] > 0) & 
#                 (test['smart_197raw'] > 0) & 
#                 (test['smart_198raw'] > 0)]  # 筛选raw:5,187,188,197,198任一大于0
print(test1.shape)  # (33111, 49)
tg = test1.groupby(["manufacturer", "model", "serial_number"])['dt'].min().reset_index()
print(tg.shape)  # 2213，11
tg.head()


# In[16]:


# test划分为8份数据
test = pd.read_csv(path + '/data/processed_date_data/test.csv')
print(test.shape)  # (178028, 49)
test1 = test[(test['smart_5raw'] > 0) | 
                (test['smart_187raw'] > 0) | 
                (test['smart_188raw'] > 0) | 
                (test['smart_197raw'] > 0) | 
                (test['smart_198raw'] > 0)]  # 筛选raw:5,187,188,197,198任一大于0
print(test1.shape)  # (33111, 49)
test11 = test1[test1['smart_9raw'] <= 10000]
test12 = test1[(test1['smart_9raw'] > 10000) & (test1['smart_9raw'] <= 20000)]
test13 = test1[(test1['smart_9raw'] > 20000) & (test1['smart_9raw'] <= 30000)]
test14 = test1[(test1['smart_9raw'] > 30000)]
print(test11.shape, test12.shape, test13.shape, test14.shape)

test0 = test[(test['smart_5raw']==0) & 
                (test['smart_187raw']==0) & 
                (test['smart_188raw']==0) & 
                (test['smart_197raw']==0) & 
                (test['smart_198raw']==0)]  # 筛选raw:5,187,188,197,198全为0
print(test0.shape)  # (144915, 49)
test01 = test0[test0['smart_9raw'] <= 10000]
test02 = test0[(test0['smart_9raw'] > 10000) & (test0['smart_9raw'] <= 20000)]
test03 = test0[(test0['smart_9raw'] > 20000) & (test0['smart_9raw'] <= 30000)]
test04 = test0[(test0['smart_9raw'] > 30000)]
print(test01.shape, test02.shape, test03.shape, test04.shape)


# In[18]:


test11.to_csv(path + '/data/processed_date_data/test11.csv', index=False)
test12.to_csv(path + '/data/processed_date_data/test12.csv', index=False)
test13.to_csv(path + '/data/processed_date_data/test13.csv', index=False)
test14.to_csv(path + '/data/processed_date_data/test14.csv', index=False)
test01.to_csv(path + '/data/processed_date_data/test01.csv', index=False)
test02.to_csv(path + '/data/processed_date_data/test02.csv', index=False)
test03.to_csv(path + '/data/processed_date_data/test03.csv', index=False)
test04.to_csv(path + '/data/processed_date_data/test04.csv', index=False)


# # 负样本采样：在训练集中，以各条件下的正样本数×10的数量对负样本采样 同时每个条件下的负样本按月均匀分布 如果前面的月份数量不够可用后面的月份补

# In[ ]:


# 正样本8类大小：(3086, 52) (3655, 52) (2816, 52) (1455, 52),(8105, 52) (4662, 52) (4324, 52) (1885, 52)
# 负样本：12个月份


# In[19]:


tf11_num = (traint11.shape[0] * 10) // 12
tf12_num = (traint12.shape[0] * 10) // 12
tf13_num = (traint13.shape[0] * 10) // 12
tf14_num = (traint14.shape[0] * 10) // 12
tf01_num = (traint01.shape[0] * 10) // 12
tf02_num = (traint02.shape[0] * 10) // 12
tf03_num = (traint03.shape[0] * 10) // 12
tf04_num = (traint04.shape[0] * 10) // 12
print(tf11_num,tf12_num,tf13_num,tf14_num,tf01_num,tf02_num,tf03_num,tf04_num)


# In[26]:


# 抽取异常数据中前4类数据：11，12，13，14
print('tf11_num,tf12_num,tf13_num,tf14_num:',tf11_num,tf12_num,tf13_num,tf14_num)
tf11, tf12, tf13, tf14 = [], [], [], []
path_files = path + '/data/processed_date_data/'
del_cols = ['smart_10raw','smart_10_normalized','smart_240raw','smart_240_normalized',
           'smart_241raw','smart_241_normalized','smart_242raw','smart_242_normalized']
na11, na12, na13, na14 = 0, 0, 0, 0
files = ['1801','1802','1803','1804','1805','1806','1707','1708','1709','1710','1711','1712']
ff = ['20' + x + '.csv' for x in files]
#for file in os.listdir(path_files):
for file in ff:
    if file[-3:] == 'csv' and file[:2] == '20':
        n11, n12, n13, n14 = tf11_num + na11, tf12_num + na12, tf13_num + na13, tf14_num + na14
        tp = pd.read_csv(path_files+file)
        print(file, tp.shape)
        tp = tp[set(tp.columns)-set(del_cols)]  # 删除8个列：raw+normalized:10，240，241，242
        tp = tp[tp['model'] == 1]  # 筛选model==1
        tp = tp[tp['label'] == 0]  # 筛选出负样本
        tpp = tp[(tp['smart_5raw'] > 0) | 
                (tp['smart_187raw'] > 0) | 
                (tp['smart_188raw'] > 0) | 
                (tp['smart_197raw'] > 0) | 
                (tp['smart_198raw'] > 0)]  # 筛选raw:5,187,188,197,198任一为0
        tpp.reset_index(drop=True, inplace=True)  # 一定要重新设定index
        print(tpp.shape)
        tp11 = tpp[tpp['smart_9raw'] <= 10000]
        tp12 = tpp[(tpp['smart_9raw'] > 10000) & (tpp['smart_9raw'] <= 20000)]
        tp13 = tpp[(tpp['smart_9raw'] > 20000) & (tpp['smart_9raw'] <= 30000)]
        tp14 = tpp[(tpp['smart_9raw'] > 30000)]
        if tp11.shape[0] < n11:
            na11 = n11 - tp11.shape[0]
            n11 = tp11.shape[0]
        else:
            na11 = 0
        print('tf11_num, n11, na11, tp11.shape[0]:',tf11_num, n11, na11, tp11.shape[0])
        if tp12.shape[0] < n12:
            na12 = n12 - tp12.shape[0]
            n12 = tp12.shape[0]
        else:
            na12 = 0
        print('tf12_num, n12, na12, tp12.shape[0]:',tf12_num, n12, na12, tp12.shape[0])
        if tp13.shape[0] < n13:
            na13 = n13 - tp13.shape[0]
            n13 = tp13.shape[0]
        else:
            na13 = 0
        print('tf13_num, n13, na13, tp13.shape[0]:',tf13_num, n13, na13, tp13.shape[0])
        if tp14.shape[0] < n14:
            na14 = n14 - tp14.shape[0]
            n14 = tp14.shape[0]
        else:
            na14 = 0
        print('tf14_num, n14, na14, tp14.shape[0]:',tf14_num, n14, na14, tp14.shape[0])
        ts11 = tp11.sample(n=n11)
        ts12 = tp12.sample(n=n12)
        ts13 = tp13.sample(n=n13)
        ts14 = tp14.sample(n=n14)
        tf11.append(ts11)
        tf12.append(ts12)
        tf13.append(ts13)
        tf14.append(ts14)
trtf11 = pd.concat(tf11, ignore_index=True)
trtf12 = pd.concat(tf12, ignore_index=True)
trtf13 = pd.concat(tf13, ignore_index=True)
trtf14 = pd.concat(tf14, ignore_index=True)
print(trtf11.shape, trtf12.shape, trtf13.shape, trtf14.shape)  # (28403, 44) (36540, 44) (28152, 44) (14544, 44)
trtf11.to_csv(path + '/data/processed_date_data/trainf11.csv', index=False)
trtf12.to_csv(path + '/data/processed_date_data/trainf12.csv', index=False)
trtf13.to_csv(path + '/data/processed_date_data/trainf13.csv', index=False)
trtf14.to_csv(path + '/data/processed_date_data/trainf14.csv', index=False)


# In[28]:


# 抽取非异常数据中后4类数据: 01，02，03，04
print('tf01_num,tf02_num,tf03_num,tf04_num:',tf01_num,tf02_num,tf03_num,tf04_num)
tf01, tf02, tf03, tf04 = [], [], [], []
path_files = path + '/data/processed_date_data/'
del_cols = ['smart_10raw','smart_10_normalized','smart_240raw','smart_240_normalized',
           'smart_241raw','smart_241_normalized','smart_242raw','smart_242_normalized']
na01, na02, na03, na04 = 0, 0, 0, 0
for file in os.listdir(path_files):
    if file[-3:] == 'csv' and file[:2] == '20':
        n01, n02, n03, n04 = tf01_num + na01, tf02_num + na02, tf03_num + na03, tf04_num + na04
        tp = pd.read_csv(path_files+file)
        print(file, tp.shape)
        tp = tp[set(tp.columns)-set(del_cols)]  # 删除8个列：raw+normalized:10，240，241，242
        tp = tp[tp['model'] == 1]  # 筛选model==1
        tp = tp[tp['label'] == 0]  # 筛选出负样本
        tpp = tp[(tp['smart_5raw'] == 0) & 
                (tp['smart_187raw'] == 0) & 
                (tp['smart_188raw'] == 0) & 
                (tp['smart_197raw'] == 0) & 
                (tp['smart_198raw'] == 0)]  # 筛选raw:5,187,188,197,198全为0
        tpp.reset_index(drop=True, inplace=True)  # 一定要重新设定index
        print(tpp.shape)
        tp01 = tpp[tpp['smart_9raw'] <= 10000]
        tp02 = tpp[(tpp['smart_9raw'] > 10000) & (tpp['smart_9raw'] <= 20000)]
        tp03 = tpp[(tpp['smart_9raw'] > 20000) & (tpp['smart_9raw'] <= 30000)]
        tp04 = tpp[(tpp['smart_9raw'] > 30000)]
        if tp01.shape[0] < n01:
            na01 = n01 - tp01.shape[0]
            n01 = tp01.shape[0]
        else:
            na01 = 0
        print('tf01_num, n01, na01, tp01.shape[0]:',tf01_num, n01, na01, tp01.shape[0])
        if tp02.shape[0] < n02:
            na02 = n02 - tp02.shape[0]
            n02 = tp02.shape[0]
        else:
            na02 = 0
        print('tf02_num, n02, na02, tp02.shape[0]:',tf02_num, n02, na02, tp02.shape[0])
        if tp03.shape[0] < n03:
            na03 = n03 - tp03.shape[0]
            n03 = tp03.shape[0]
        else:
            na03 = 0
        print('tf03_num, n03, na03, tp03.shape[0]:',tf03_num, n03, na03, tp03.shape[0])
        if tp04.shape[0] < n04:
            na14 = n04 - tp04.shape[0]
            n04 = tp04.shape[0]
        else:
            na04 = 0
        print('tf04_num, n04, na04, tp04.shape[0]:',tf04_num, n04, na04, tp04.shape[0])
        ts01 = tp01.sample(n=n01)
        ts02 = tp02.sample(n=n02)
        ts03 = tp03.sample(n=n03)
        ts04 = tp04.sample(n=n04)
        tf01.append(ts01)
        tf02.append(ts02)
        tf03.append(ts03)
        tf04.append(ts04)
trtf01 = pd.concat(tf01, ignore_index=True)
trtf02 = pd.concat(tf02, ignore_index=True)
trtf03 = pd.concat(tf03, ignore_index=True)
trtf04 = pd.concat(tf04, ignore_index=True)
print(trtf01.shape, trtf02.shape, trtf03.shape, trtf04.shape)  # (81048, 44) (46620, 44) (43236, 44) (18840, 44)
trtf01.to_csv(path + '/data/processed_date_data/trainf01.csv', index=False)
trtf02.to_csv(path + '/data/processed_date_data/trainf02.csv', index=False)
trtf03.to_csv(path + '/data/processed_date_data/trainf03.csv', index=False)
trtf04.to_csv(path + '/data/processed_date_data/trainf04.csv', index=False)


# In[3]:


# 用disk_first.csv每个硬盘观察的第一天与正负样本每行匹配获得每个硬盘当时的观察天数：dt-disk_first.dt
dt = pd.read_csv(path +'/data/processed_date_data/disk1_dt_first.csv')


# In[37]:


# 8类正样本构建硬盘观察天数
# 正样本8类大小：(3086, 52) (3655, 52) (2816, 52) (1455, 52),(8105, 52) (4662, 52) (4324, 52) (1885, 52)
tt = pd.read_csv(path + '/data/processed_date_data/traint11.csv')
print(tt.shape)  # (7505, 44)
tt = tt.merge(dt, how='left',on='serial_number')
tt['days'] = (pd.to_datetime(tt['dt'])-pd.to_datetime(tt['dt_first'])).dt.days
tt.to_csv(path + '/data/processed_date_data/traintd11.csv', index=False)
print(tt.shape)  # (7505, 46)
tt.head(2)


# In[45]:


# 8类负样本构建硬盘观察天数
# 负样本8类大小：(28403, 44) (36540, 44) (28152, 44) (14544, 44)，(81048, 44) (46620, 44) (43236, 44) (18840, 44)
tt = pd.read_csv(path + '/data/processed_date_data/trainf04.csv')
print(tt.shape)  
tt = tt.merge(dt, how='left',on='serial_number')
tt['days'] = (pd.to_datetime(tt['dt'])-pd.to_datetime(tt['dt_first'])).dt.days
tt.to_csv(path + '/data/processed_date_data/trainfd04.csv', index=False)
print(tt.shape)  
tt.head(2)


# In[11]:


# 8类测试集构建硬盘观察天数
# 测试集8类大小：(89, 49) (3416, 49) (15442, 49) (14164, 49),(960, 49) (23532, 49) (58286, 49) (62137, 49)
tt = pd.read_csv(path + '/data/processed_date_data/test01.csv')
print(tt.shape)  
tt = tt.merge(dt, how='left',on='serial_number')
tt['days'] = (pd.to_datetime(tt['dt'])-pd.to_datetime(tt['dt_first'])).dt.days
tt.to_csv(path + '/data/processed_date_data/testd01.csv', index=False)
print(tt.shape)  
tt.head(2)


# In[ ]:




