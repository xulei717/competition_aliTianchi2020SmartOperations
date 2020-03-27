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
from feature import data_cleaning
cleaning = data_cleaning.DataCleaning()

# # 训练集初始筛选特征，清洗后存入user_data/tmp_data文件里
# cleaning.dataclean()
#
# # 给训练集数据打标签
# print('给训练集打标签～')
# from feature import label
# labeling = label.Label()
# labeling.get_label()

# 筛选出测试集中唯一值特征，并在测试集和训练集中删除这些唯一值特征
cleaning.read_data()


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
for f in ['user_data/tmp_data/201806.csv', 'user_data/tmp_data/201807.csv']:
    print(f)
    df = pd.read_csv(f)
    # df = df[df.model == 2]
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

features = [x for x in train.columns if x not in ["serial_number", "manufacturer", "model", "dt", "label",
                                                  "smart_5_normalized", "smart_187_normalized", "smart_188_normalized",
                                                  "smart_197_normalized", "smart_198_normalized"]]
target = 'label'
from model import basic_model
model = basic_model.BasicModel()

def model_result(model_number):
    pos0 = pos[pos.model==model_number]
    neg0 = neg[neg.model==model_number]
    train0 = pd.concat([pos0, neg0])
    test0 = test1[test1.model==model_number]
    print(model_number, '-train: ', train0[features].shape, pos0[features].shape, neg0[features].shape, 'test: ', test0[features].shape)
    result1 = model.lightgbm_model(train0, test0, features, target)
    print('result1: ', result1.shape)
    return result1

result11 = model_result(1)
result12 = model_result(2)
result2 = test[(test.smart_5_normalized < 100)|(test.smart_187_normalized < 100)|(test.smart_188_normalized < 100)|
               (test.smart_197_normalized < 100)|(test.smart_198_normalized < 100)]
# result2 = result2[(result2.smart_5raw > 0)|((result2.smart_187raw > 0)|(result2.smart_188raw > 0))]  # registry.cn-shanghai.aliyuncs.com/xl717/pakdd2020:2.3
result2 = result2[(result2.smart_5raw > 0)&((result2.smart_187raw > 0)|(result2.smart_188raw > 0))]  # registry.cn-shanghai.aliyuncs.com/xl717/pakdd2020:2.4
result2 = result2.groupby(["manufacturer", "model", "serial_number"])["dt"].max().reset_index()
result = pd.concat([result11, result12, result2])
print('result: ', result.shape, 'result2: ', result2.shape, 'result11: ', result11.shape, 'result12: ', result12.shape)
# test = test.groupby(["manufacturer", "model", "serial_number"])["dt"].max().reset_index()
# result = result[["manufacturer", "model", "serial_number"]].merge(test, on=["manufacturer", "model", "serial_number"], how="left")
result_path = 'prediction_result/predictions.csv'
result.to_csv(result_path, index=False, header=None)
print(result.shape)

# 把结果压缩到result.zip文件
import zipfile

with zipfile.ZipFile('result.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(result_path, 'result.csv')