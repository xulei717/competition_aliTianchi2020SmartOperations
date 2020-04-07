# -*- coding:utf-8 -*-

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
# # # 训练集初始筛选特征，清洗后存入user_data/tmp_data文件里
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

# # # 给训练集数据打标签
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

# 筛选出测试集中唯一值特征，并在测试集和训练集中删除这些唯一值特征
year_month = '201809'  # docker中需要换成'201809'
train_names = ['201707.csv', '201708.csv', '201709.csv', '201710.csv', '201711.csv', '201712.csv',
            '201801.csv', '201802.csv', '201803.csv', '201804.csv', '201805.csv', '201806.csv',
            '201807.csv', '201808.csv']
train_files = [tmp_path+x for x in train_names]
cleaning.read_data(year_month, train_files)


# 导入特征工程
print("导入特征工程～")
from feature import feature
feature_class = feature.Feature()


# 预测输出结果
print("预测输出结果～")
test = pd.read_csv('user_data/tmp_data/test.csv', index_col=None)
print('test: ', test.shape)
test = feature_class.feature_extraction(test)
print('test: ', test.shape)

train_list = list()
train_paths = ['user_data/tmp_data/' + x for x in train_names]
for f in train_paths:
    print(f)
    df = pd.read_csv(f)
    df = feature_class.feature_extraction(df)
    train_list.append(df)
train = pd.concat(train_list)
print('train: ', train.shape)

pos = train[train.label == 1]
neg = train[train.label == 0]
print('pos: ', pos.shape, 'neg: ', neg.shape)
neg = neg.sample(n=pos.shape[0]*10, random_state=2020)
print('neg: ', neg.shape)

features = [x for x in train.columns if x not in ["serial_number", "manufacturer", "model", "dt", "label"]]
target = 'label'
print('features: ', len(features), features)


from model import basic_model
model = basic_model.BasicModel()

def model_result(model_number):
    pos0 = pos[pos.model==model_number]
    neg0 = neg[neg.model==model_number]
    train0 = pd.concat([pos0, neg0])
    test0 = test[test.model==model_number]
    print(model_number, '-train: ', train0[features].shape, pos0[features].shape, neg0[features].shape, 'test: ', test0[features].shape)
    result0 = model.lightgbm_model(train0, test0, features, target)
    print('result0: ', result0.shape)
    return result0

result1 = model_result(1)
result2 = model_result(2)
result = pd.concat([result1, result2])
print('result: ', result.shape, 'result1: ', result1.shape, 'result2: ', result2.shape)

# train0 = pd.concat([pos, neg])
# print('-train: ', train0[features].shape, 'test: ',
#       test[features].shape)
# result = model.lightgbm_model(train0, test, features, target)
# print('result: ', result.shape)

result = result.drop_duplicates(['model', 'serial_number'], keep='last')
print('result drop_duplicates: ', result.shape)
result_path = 'prediction_result/predictions.csv'
result.to_csv(result_path, index=False, header=None)
print(result.shape)

# 把结果压缩到result.zip文件
import zipfile

with zipfile.ZipFile('result.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(result_path, 'result.csv')