# -*- coding:utf-8 -*-

import json
import pandas as pd
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
parentPath = os.path.split(curPath)[0]
rootPath = os.path.split(parentPath)[0]
print(curPath)
print(parentPath)
sys.path.append(parentPath)

# 可调节的参数
# 特征工程：是否构建diff特征，是否新增raw/normalized列，是否新增raw/normalized的diff特征，是：1，否：0
# is_raw_normalized_diff在is_raw_normalized为1的前提下，才可以赋值为1
is_getdiff, is_raw_normalized, is_raw_normalized_diff = 1, 1, 0
# 特征工程：是否保留raw特征，是否保留normalized列，是否保留diff特征，是：1，否：0
# is_deldiff在is_getdiff，is_raw_normalized_diff至少有一个为1的前期下，才可以赋值为1
is_delraw, is_delnormalized, is_deldiff = 0, 0, 0
# 数据筛选：训练集中负样本只保留正样本的前20天数据，测试集中样本只保留样本的最近21天数据，同为1或着同为0，测试集和训练集保持一致
# 数据筛选必须在diff构建之后执行
is_data_neg, is_test21 = 0, 0

# lightgbm模型的折数和预测阈值
n_splits, pred_threshold = 10, 0.5

train_path = 'data/round2_train/'
test_path = 'tcdata/disk_sample_smart_log_round2/*'
tmp_path = 'user_data/tmp_data/'

# # # 数据清洗
print("开始数据清洗～")
from feature import data_cleaning
cleaning = data_cleaning.DataCleaning(train_path, test_path, tmp_path)

# ----------------------------------线下提前做好--------------------------------------------------
# # 训练集初始筛选特征，清洗后存入user_data/tmp_data文件里
# cleaning.dataclean(train_path + 'disk_sample_smart_log_201707.csv')
# cleaning.dataclean(train_path + 'disk_sample_smart_log_201708.csv')
# cleaning.dataclean(train_path + 'disk_sample_smart_log_201709.csv')
# cleaning.dataclean(train_path + 'disk_sample_smart_log_201710.csv')
# cleaning.dataclean(train_path + 'disk_sample_smart_log_201711.csv')
# cleaning.dataclean(train_path + 'disk_sample_smart_log_201712.csv')
# cleaning.dataclean(train_path + 'disk_sample_smart_log_201801.csv')
# cleaning.dataclean(train_path + 'disk_sample_smart_log_201802.csv')
# cleaning.dataclean(train_path + 'disk_sample_smart_log_201803.csv')
# cleaning.dataclean(train_path + 'disk_sample_smart_log_201804.csv')
# cleaning.dataclean(train_path + 'disk_sample_smart_log_201805.csv')
# cleaning.dataclean(train_path + 'disk_sample_smart_log_201806.csv')
# cleaning.dataclean(train_path + 'disk_sample_smart_log_201807.csv')
# cleaning.dataclean(train_path + 'disk_sample_smart_log_201808.csv')
# sys.exit()

# # 给训练集数据打标签
# print('给训练集打标签～')
# from feature import label
# labeling = label.Label()
# tag_path = 'data/round2_train/disk_sample_fault_tag.csv'
# for year_month in ['201707', '201708', '201709', '201710', '201711', '201712', '201801', '201802', '201803', '201804',
#                    '201805', '201806', '201807']:
#     data_path = tmp_path + year_month + '.csv'
#     labeling.get_label(tag_path, data_path, year_month)
# # 201808
# tag_path = 'data/round2_train/disk_sample_fault_tag_201808.csv'
# data_path = tmp_path + '201808.csv'
# labeling.get_label(tag_path, data_path, '201808')
# sys.exit()

# ---------------------------------------线上执行------------------------------------
# # 筛选出测试集中唯一值特征，并在测试集和训练集中删除这些唯一值特征
# year_month = '201808'  # docker中需要换成'201809'
year_month = '201809'  # docker中需要换成'201809'
train_names = ['201707.csv', '201708.csv', '201709.csv', '201710.csv', '201711.csv', '201712.csv',
               '201801.csv', '201802.csv', '201803.csv', '201804.csv', '201805.csv', '201806.csv', '201807.csv'
               # ]
               , '201808.csv']
train_files = [tmp_path+x for x in train_names]
cleaning.read_data(year_month, train_files)  # test:82-34, train:35


# 导入特征工程
print("导入特征工程～")
from feature import feature
feature_class = feature.Feature()

# 特征工程：删除列，构建diff特征，构建raw/normalized特征，构建raw/normalized的diff特征,删除列
test = pd.read_csv(tmp_path + 'test.csv', index_col=None)
print('test: ', test.shape)
test = feature_class.feature_del(test)
print('test feature_del: ', test.shape)
if is_getdiff:
    test = feature_class.feature_diff(test)
    print('test feature_diff: ', test.shape)
if is_raw_normalized:
    test = feature_class.feature_raw_normalized(test)
    print('test feature_raw_normalized: ', test.shape)
if is_raw_normalized_diff:
    test = feature_class.feature_raw_normalized_diff(test)
    print('test feature_raw_normalized_diff: ', test.shape)
if is_test21:
    test = cleaning.test21(test)
    print('test is_test21: ', test.shape)
if is_delraw:
    test = feature_class.feature_delraw(test)
    print('test feature_delraw: ', test.shape)
if is_delnormalized:
    test = feature_class.feature_delnormalized(test)
    print('test feature_delnormalized: ', test.shape)
if is_deldiff:
    test = feature_class.feature_deldiff(test)
    print('test feature_deldiff: ', test.shape)

train_pos_list = list()
train_neg_list = list()
train_paths = ['user_data/tmp_data/' + x for x in train_names]
for f in train_paths:
    print(f)
    df = pd.read_csv(f)
    df = feature_class.feature_del(df)
    if is_getdiff:
        df = feature_class.feature_diff(df)
        print('train feature_diff: ', df.shape)
    if is_raw_normalized:
        df = feature_class.feature_raw_normalized(df)
        print('train feature_raw_normalized: ', df.shape)
    if is_raw_normalized_diff:
        df = feature_class.feature_raw_normalized_diff(df)
        print('train feature_raw_normalized_diff: ', df.shape)
    pos = df[df.label == 1]
    neg = df[df.label == 0]
    if is_data_neg:
        neg = cleaning.data_neg(neg)
        print('train neg is_data_neg: ', neg.shape)
    neg = neg.sample(n=pos.shape[0]*10, random_state=2020)
    train_pos_list.append(pos)
    train_neg_list.append(neg)
train_pos = pd.concat(train_pos_list)
train_neg = pd.concat(train_neg_list)
print('train_pos: ', train_pos.shape, 'train_neg: ', train_neg.shape)

train = pd.concat([train_pos, train_neg])
if is_delraw:
    train = feature_class.feature_delraw(train)
    print('train feature_delraw: ', train.shape)
if is_delnormalized:
    train = feature_class.feature_delnormalized(train)
    print('train feature_delnormalized: ', train.shape)
if is_deldiff:
    train = feature_class.feature_deldiff(train)
    print('train feature_deldiff: ', train.shape)
print(len(train.columns), train.columns)


# 训练模型预测结果
test = test.sort_values(['model', 'serial_number', 'dt'])
test = test.drop_duplicates(['model', 'serial_number'], keep='last')
print('test drop_duplicates last: ', test.shape)
features = [x for x in train.columns if x not in ["serial_number", "manufacturer", "model", "dt", "label"]]
target = 'label'
print('features: ', len(features), features)
print('train: ', train[features].shape, 'test: ', test[features].shape)


from model import basic_model
model = basic_model.BasicModel()

# def model_result(model_number):
#     pos0 = pos[pos.model==model_number]
#     neg0 = neg[neg.model==model_number]
#     train0 = pd.concat([pos0, neg0])
#     test0 = test[test.model==model_number]
#     print(model_number, '-train: ', train0[features].shape, pos0[features].shape, neg0[features].shape, 'test: ', test0[features].shape)
#     result0 = model.lightgbm_model(train0, test0, features, target)
#     print('result0: ', result0.shape)
#     return result0

# result1 = model_result(1)
# result2 = model_result(2)
# result = pd.concat([result1, result2])
# print('result: ', result.shape, 'result1: ', result1.shape, 'result2: ', result2.shape)

result = model.lightgbm_model(train, test, features, target, n_splits, pred_threshold)
print('result: ', result.shape)

result = result.drop_duplicates(['model', 'serial_number'], keep='last')
print('result drop_duplicates: ', result.shape)
result_path = 'prediction_result/predictions.csv'
result.to_csv(result_path, index=False, header=None)
print(result.shape)

# 把结果压缩到result.zip文件
import zipfile

with zipfile.ZipFile('result.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(result_path, 'result.csv')
