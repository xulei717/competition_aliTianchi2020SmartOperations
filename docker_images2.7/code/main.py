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

train_path = 'data/round2_train/'
test_path = 'tcdata/disk_sample_smart_log_round2/*'
tmp_path = 'user_data/tmp_data/'

# # # 数据清洗
print("开始数据清洗～")
from feature import data_cleaning
cleaning = data_cleaning.DataCleaning(train_path, test_path, tmp_path)
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
# sys.exit()

# 导入特征工程
print("导入特征工程～")
from feature import feature
feature_class = feature.Feature()

# 预测输出结果
print("预测输出结果～")
test = pd.read_csv('user_data/tmp_data/test.csv', index_col=None)
print('test: ', test.shape)
test = feature_class.feature_extraction(test)
print('test: ', test.shape)  # (4985368, 34)
# test.fillna(0, inplace=True)
test.to_csv(tmp_path + 'test.csv', index=False)

train_pos_list = list()
train_neg_list = list()
train_paths = ['user_data/tmp_data/' + x for x in train_names]
for f in train_paths:
    print(f)
    df = pd.read_csv(f)
    df = feature_class.feature_extraction(df)
    df.to_csv(f, index=False)
    pos = df[df.label == 1]
    neg = df[df.label == 0]
    neg = neg.sample(n=pos.shape[0]*10, random_state=2020)
    train_pos_list.append(pos)
    train_neg_list.append(neg)
train_pos = pd.concat(train_pos_list)
train_neg = pd.concat(train_neg_list)
print('train_pos: ', train_pos.shape, 'train_neg: ', train_neg.shape)


train = pd.concat([train_pos, train_neg])
# train0.fillna(0, inplace=True)
train.to_csv(tmp_path + 'train.csv', index=False)


# 训练模型预测结果
train = pd.read_csv(tmp_path + 'train.csv', index_col=None)
print('train: ', train.shape)
test = pd.read_csv(tmp_path + 'test.csv', index_col=None)
print('test: ', test.shape)
test = test.sort_values(['model', 'serial_number', 'dt'])
test = test.drop_duplicates(['model', 'serial_number'], keep='last')
print('test: ', test.shape)
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

result = model.lightgbm_model(train, test, features, target)
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