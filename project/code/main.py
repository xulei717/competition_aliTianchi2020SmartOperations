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


# 数据清洗
print("开始数据清洗～")
from feature import data_cleaning, label
cleaning = data_cleaning.DataCleaning()
cleaning.read_data()
labeling = label.Label()
labeling.get_label()


# 导入特征工程
print("导入特征工程～")
from feature import feature
feature_class = feature.Feature()


# 预测输出结果
print("预测输出结果～")
test = pd.read_csv('user_data/tmp_data/test_b.csv', index_col=None)
print('test: ', test.shape)
test = feature_class.feature_extraction(test)
print('test: ', test.shape)

train_list = list()
for f in ['user_data/tmp_data/201805.csv', 'user_data/tmp_data/201806.csv']:
    print(f)
    df = pd.read_csv(f)
    df = df[df.model == 2]
    df = feature_class.feature_extraction(df)
    train_list.append(df)
train = pd.concat(train_list)
print('train: ', train.shape)

test1 = test[(test.smart_5_normalized == 100) &
            (test.smart_187_normalized == 100) &
            (test.smart_188_normalized == 100) &
            (test.smart_197_normalized == 100) &
            (test.smart_198_normalized == 100)]

pos = train[(train.label == 1) &
            (train.smart_5_normalized == 100) &
            (train.smart_187_normalized == 100) &
            (train.smart_188_normalized == 100) &
            (train.smart_197_normalized == 100) &
            (train.smart_198_normalized == 100)]
neg = train[(train.label == 0) &
            (train.smart_5_normalized == 100) &
            (train.smart_187_normalized == 100) &
            (train.smart_188_normalized == 100) &
            (train.smart_197_normalized == 100) &
            (train.smart_198_normalized == 100)]
print('pos: ', pos.shape)
neg = neg.sort_values("dt").drop_duplicates(["manufacturer", "model", "serial_number"], keep="last")
print('neg: ', neg.shape)

# neg = neg.sample(frac=0.5, random_state=2020)
# print('neg: ', neg.shape)

train = pd.concat([pos, neg])
features = [x for x in train.columns if x not in ["serial_number", "manufacturer", "model", "dt", "label",
                                                  "smart_5_normalized", "smart_187_normalized", "smart_188_normalized",
                                                  "smart_197_normalized", "smart_198_normalized"]]
print('train: ', train[features].shape, 'test: ', test[features].shape)
target = 'label'


from model import basic_model

result_path = 'prediction_result/predictions.csv'
model = basic_model.BasicModel()
result1 = model.lightgbm_model(train, test1, features, target)

result2 = test[(test.smart_5_normalized < 100)|(test.smart_187_normalized < 100)|(test.smart_188_normalized < 100)|(test.smart_197_normalized < 100)|(test.smart_198_normalized < 100)]
result2 = result2[(result2.smart_5raw > 0)|((result2.smart_187raw > 0)|(result2.smart_188raw > 0))]
result2 = result2.groupby(["manufacturer", "model", "serial_number"])["dt"].max().reset_index()
result = pd.concat([result1, result2])
# test = test.groupby(["manufacturer", "model", "serial_number"])["dt"].max().reset_index()
# result = result[["manufacturer", "model", "serial_number"]].merge(test, on=["manufacturer", "model", "serial_number"], how="left")
result.to_csv(result_path, index=False, header=None)
print(result.shape)