"""
@author: Yang Zhuoshi
@email: yangzhuoshi@boe.com.cn
@file_name: process.py
@create_time: 2020/3/30 上午10:37
"""
import sys
import pandas as pd
from collections import Counter


def cal_score(merge_data):

    tp = len(merge_data[(merge_data['label'] == 1) & (merge_data['pre_label'] == 1)])
    fp = len(merge_data[(merge_data['pre_label'] == 1) & (merge_data['label'] == 0)])
    fn = len(merge_data[(merge_data['pre_label'] == 0) & (merge_data['label'] == 1)])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return tp, fp, fn, precision, recall, f1


def read_data_get_label():
    ground_truth = pd.read_csv('../0326/project/user_data/tmp_data/tag.csv')
    ground_truth['label'] = 1
    ground_truth['model'] = ground_truth['model'].astype(int)
    ground_truth['model'] = ground_truth['model'].astype(str)
    ground_truth['key'] = ground_truth['model'] + ground_truth['serial_number']
    ground_truth['fault_time'] = pd.to_datetime(ground_truth['fault_time'])
    prediction_list = ['../0326/project/user_data/tmp_data/predictions_a.csv',
                       '../0326/project/user_data/tmp_data/predictions_b.csv']
    for item in prediction_list:
        prediction = pd.read_csv(item, names=['manufacturer', 'model', 'serial_number', 'dt'])
        prediction['model'] = prediction['model'].astype(int)
        prediction['model'] = prediction['model'].astype(str)
        prediction['key'] = prediction['model'] + prediction['serial_number']
        prediction['dt'] = pd.to_datetime(prediction['dt'])
        data = prediction.merge(ground_truth, on=['serial_number', 'model', 'key'], how='outer')
        data['pre_label'] = 0
        # data['fault_time'] = data['fault_time'].fillna('2020-03-30')
        # data['fault_time'] = pd.to_datetime(data['fault_time'])
        # data.loc[((data['fault_time'] - data['dt']).dt.days >= 0) & ((data['fault_time'] - data['dt']).dt.days <= 30), 'pre_label'] = 1
        data['label'] = data['label'].fillna(0)
        data.loc[(~data['fault_time'].isna()) & (~data['dt'].isna()), 'pre_label'] = 1
        tp, fp, fn, precision, recall, f1 = cal_score(data)
        print(item)
        # print('---------------------------------')
        print('tp:' + str(tp))
        print('fp:' + str(fp))
        print('fn:' + str(fn))
        print('precision:' + str(precision))
        print('recall:' + str(recall))
        print('f1 score:' + str(f1))
        print('---------------------------------')
    sys.exit(0)


def check_201808():
    tag = pd.read_csv('../data/round2/disk_sample_fault_tag_201808.csv')
    print(tag.shape)
    data_a = pd.read_csv('../0326/project/data/round1_testB/disk_sample_smart_log_test_a.csv')
    data_b = pd.read_csv('../0326/project/data/round1_testB/disk_sample_smart_log_test_b.csv')
    data = pd.concat([data_a, data_b])
    data['dt'] = data['dt'].apply(lambda x: str(x)[0:4] + '-' + str(x)[4:6] + '-' + str(x)[6:])
    data['dt'] = pd.to_datetime(data['dt'])
    data = data.sort_values(['dt'])
    data = data.drop_duplicates(['serial_number', 'model'], keep='last')
    print(data.shape)
    data = tag.merge(data, on=['serial_number', 'model'], how='inner')
    data['fault_time'] = pd.to_datetime(data['fault_time'])
    data['differ'] = (data['fault_time'] - data['dt']).dt.days
    print(data[['dt', 'fault_time']])
    print(len(data))
    print(Counter(list(data['differ'])))


if __name__ == '__main__':
    # read_data_get_label()
    check_201808()
