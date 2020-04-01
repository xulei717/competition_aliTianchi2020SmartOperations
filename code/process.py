"""
@author: Yang Zhuoshi
@email: yangzhuoshi@boe.com.cn
@file_name: process.py
@create_time: 2020/3/31 上午8:28
"""
import sys
import pandas as pd


class DataCleaning:
    """
    初步清理数据
    """

    def __init__(self):
        keep = [1, 3, 5, 184, 187, 188, 189, 192, 193, 194, 197, 198, 199,
                2, 8, 10, 11, 13, 181, 183, 191, 196, 200, 201, 202, 203, 204, 205, 206, 207,
                220, 221, 224, 225, 227, 228, 250, 254,
                7]
        keep_raw = ['smart_' + str(x) + 'raw' for x in keep]
        keep_normalized = ['smart_' + str(x) + '_normalized' for x in keep]
        self.keeps = keep_raw + keep_normalized + ['dt', "manufacturer", "model", "serial_number"]

    def data_clean(self, data_path):
        clean_list = []
        train = pd.read_csv(data_path, index_col=None, chunksize=100000)
        for chunk in train:
            chunk['dt'] = chunk['dt'].apply(lambda x: str(x)[0:4] + '-' + str(x)[4:6] + '-' + str(x)[6:])
            clean_list.append(chunk[self.keeps])
        combined = pd.concat(clean_list)
        combined.to_csv(data_path.split('_')[-1], index=False)
        print(data_path, combined.shape)


def tag_merge(tag_path, month):
    tag = pd.read_csv(tag_path)
    tag['key'] = tag['fault_time'].apply(lambda x: x.split('-')[1])
    tag = tag[tag['key'] == month]
    tag['tag'] = tag['tag'].astype(str)
    tag['fault_time'] = pd.to_datetime(tag['fault_time'])
    tag = tag.groupby(['serial_number', 'fault_time', 'model'])['tag'].apply(lambda x: '|'.join(x)).reset_index()
    return tag


def get_label(data_path, tag_path, month):
    """
    2018tag:manufacturer,model,serial_number,fault_time,tag,key
    """
    tag = tag_merge(tag_path, month)
    data = pd.read_csv(data_path)
    data['label'] = 0
    data['dt'] = pd.to_datetime(data['dt'])
    data = data.sort_values(['model', 'serial_number', 'dt'])
    print('original shape:' + str(data.shape))
    data_last = data.drop_duplicates(['model', 'serial_number'], keep='last')
    print('data last shape:' + str(data_last.shape))
    data_last = data_last.merge(tag, on=['model', 'serial_number'], how='left')
    data_last.loc[~data_last['tag'].isna(), 'label'] = 1
    print(data_last.columns)
    # sys.exit(0)
    data_last.drop(['fault_time', 'tag'], axis=1, inplace=True)
    data = pd.concat([data, data_last])
    data = data.sort_values(['model', 'serial_number', 'dt', 'label'])
    data = data.drop_duplicates(['model', 'serial_number', 'dt'], keep='last')
    print('done shape:' + str(data.shape))
    data.to_csv(data_path.split('.')[0] + '_labeled.csv', index=False)
    print(data.columns)
    return None


def test():
    data = pd.read_csv('201808_labeled.csv')

    print(data.columns)


def check_abnormal(data_path):
    """
    columns:['smart_1raw', 'smart_3raw', 'smart_5raw', 'smart_184raw',
       'smart_187raw', 'smart_188raw', 'smart_189raw', 'smart_192raw',
       'smart_193raw', 'smart_194raw', 'smart_197raw', 'smart_198raw',
       'smart_199raw', 'smart_2raw', 'smart_8raw', 'smart_10raw',
       'smart_11raw', 'smart_13raw', 'smart_181raw', 'smart_183raw',
       'smart_191raw', 'smart_196raw', 'smart_200raw', 'smart_201raw',
       'smart_202raw', 'smart_203raw', 'smart_204raw', 'smart_205raw',
       'smart_206raw', 'smart_207raw', 'smart_220raw', 'smart_221raw',
       'smart_224raw', 'smart_225raw', 'smart_227raw', 'smart_228raw',
       'smart_250raw', 'smart_254raw', 'smart_7raw', 'smart_1_normalized',
       'smart_3_normalized', 'smart_5_normalized', 'smart_184_normalized',
       'smart_187_normalized', 'smart_188_normalized', 'smart_189_normalized',
       'smart_192_normalized', 'smart_193_normalized', 'smart_194_normalized',
       'smart_197_normalized', 'smart_198_normalized', 'smart_199_normalized',
       'smart_2_normalized', 'smart_8_normalized', 'smart_10_normalized',
       'smart_11_normalized', 'smart_13_normalized', 'smart_181_normalized',
       'smart_183_normalized', 'smart_191_normalized', 'smart_196_normalized',
       'smart_200_normalized', 'smart_201_normalized', 'smart_202_normalized',
       'smart_203_normalized', 'smart_204_normalized', 'smart_205_normalized',
       'smart_206_normalized', 'smart_207_normalized', 'smart_220_normalized',
       'smart_221_normalized', 'smart_224_normalized', 'smart_225_normalized',
       'smart_227_normalized', 'smart_228_normalized', 'smart_250_normalized',
       'smart_254_normalized', 'smart_7_normalized', 'dt', 'manufacturer',
       'model', 'serial_number', 'label']
    """
    data = pd.read_csv(data_path)
    print(data_path + '开始分析!!!!!!!!!!!!!!')
    data['dt'] = pd.to_datetime(data['dt'])
    data = data.sort_values(['model', 'serial_number', 'dt'])
    data.drop_duplicates(['model', 'serial_number'], keep='last', inplace=True)
    total = len(data)
    total_positive = len(data[data['label'] == 1])
    print('数据总量:' + str(total) + '行!!!')
    print('positive数据总量:' + str(total_positive) + '行!!!')
    single_5 = len(data[
        (data.smart_5_normalized < 100) & (data.smart_187_normalized == 100) & (data.smart_188_normalized == 100) & (
                data.smart_197_normalized == 100) & (data.smart_198_normalized == 100)])
    single_5_positive = len(data[
        (data.smart_5_normalized < 100) & (data.smart_187_normalized == 100) & (data.smart_188_normalized == 100) & (
                data.smart_197_normalized == 100) & (data.smart_198_normalized == 100) & (data.label == 1)])
    if single_5 == 0:
        single_5_positive_5_percent = '无'
    else:
        single_5_positive_5_percent = (single_5_positive / single_5) * 100
    single_5_positive_total_percent = (single_5_positive / total_positive) * 100
    single_187 = len(data[
        (data.smart_5_normalized == 100) & (data.smart_187_normalized < 100) & (data.smart_188_normalized == 100) & (
                data.smart_197_normalized == 100) & (data.smart_198_normalized == 100)])
    single_187_positive = len(data[
        (data.smart_5_normalized == 100) & (data.smart_187_normalized < 100) & (data.smart_188_normalized == 100) & (
                data.smart_197_normalized == 100) & (data.smart_198_normalized == 100) & (data.label == 1)])
    if single_187 == 0:
        single_187_positive_187_percent = '无'
    else:
        single_187_positive_187_percent = (single_187_positive / single_187) * 100
    single_187_positive_total_percent = (single_187_positive / total_positive) * 100
    single_188 = len(data[
        (data.smart_5_normalized == 100) & (data.smart_187_normalized == 100) & (data.smart_188_normalized < 100) & (
                data.smart_197_normalized == 100) & (data.smart_198_normalized == 100)])
    single_188_positive = len(data[
        (data.smart_5_normalized == 100) & (data.smart_187_normalized == 100) & (data.smart_188_normalized < 100) & (
                data.smart_197_normalized == 100) & (data.smart_198_normalized == 100) & (data.label == 1)])
    if single_188 == 0:
        single_188_positive_188_percent = '无'
    else:
        single_188_positive_188_percent = (single_188_positive / single_188) * 100
    single_188_positive_total_percent = (single_188_positive / total_positive) * 100
    single_197 = len(data[
        (data.smart_5_normalized == 100) & (data.smart_187_normalized == 100) & (data.smart_188_normalized == 100) & (
                data.smart_197_normalized < 100) & (data.smart_198_normalized == 100)])
    single_197_positive = len(data[
        (data.smart_5_normalized == 100) & (data.smart_187_normalized == 100) & (data.smart_188_normalized == 100) & (
                data.smart_197_normalized < 100) & (data.smart_198_normalized == 100) & (data.label == 1)])
    if single_197 == 0:
        single_197_positive_197_percent = '无'
    else:
        single_197_positive_197_percent = (single_197_positive / single_197) * 100
    single_197_positive_total_percent = (single_197_positive / total_positive) * 100
    single_198 = len(data[
        (data.smart_5_normalized == 100) & (data.smart_187_normalized == 100) & (data.smart_188_normalized == 100) & (
                data.smart_197_normalized == 100) & (data.smart_198_normalized < 100)])
    single_198_positive = len(data[
        (data.smart_5_normalized == 100) & (data.smart_187_normalized == 100) & (data.smart_188_normalized == 100) & (
                data.smart_197_normalized == 100) & (data.smart_198_normalized < 100) & (data.label == 1)])
    if single_198 == 0:
        single_198_positive_198_percent = '无'
    else:
        single_198_positive_198_percent = (single_198_positive / single_198) * 100
    single_198_positive_total_percent = (single_198_positive / total_positive) * 100
    all_abnormal = len(data[
        (data.smart_5_normalized < 100) & (data.smart_187_normalized < 100) & (data.smart_188_normalized < 100) & (
                data.smart_197_normalized < 100) & (data.smart_198_normalized < 100)])
    all_abnormal_positive = len(data[
        (data.smart_5_normalized < 100) & (data.smart_187_normalized < 100) & (data.smart_188_normalized < 100) & (
                data.smart_197_normalized < 100) & (data.smart_198_normalized < 100) & (data.label == 1)])
    if all_abnormal == 0:
        all_abnormal_positive_all_abnormal_percent = '无'
    else:
        all_abnormal_positive_all_abnormal_percent = (all_abnormal_positive / all_abnormal) * 100
    all_abnormal_positive_total_percent = (all_abnormal_positive / total_positive) * 100
    any_abnormal = len(data[
        (data.smart_5_normalized < 100) | (data.smart_187_normalized < 100) | (data.smart_188_normalized < 100) | (
                data.smart_197_normalized < 100) | (data.smart_198_normalized < 100)])
    any_abnormal_positive = len(data[
                                    ((data.smart_5_normalized < 100) | (data.smart_187_normalized < 100) | (data.smart_188_normalized < 100) | (
                data.smart_197_normalized < 100) | (data.smart_198_normalized < 100)) & (data.label == 1)])
    if any_abnormal == 0:
        any_abnormal_positive_any_abnormal_percent = '无'
    else:
        any_abnormal_positive_any_abnormal_percent = (any_abnormal_positive / any_abnormal) * 100
    any_abnormal_positive_total_percent = (any_abnormal_positive / total_positive) * 100
    print('只有5异常: ' + str(single_5))
    print('只有5异常且label为1: ' + str(single_5_positive))
    print('只有5异常且label为1的在5异常中的占比: ' + str(single_5_positive_5_percent))
    print('只有5异常且label为1的在全部label为1中的占比: ' + str(single_5_positive_total_percent))
    print('只有187异常: ' + str(single_187))
    print('只有187异常且label为1: ' + str(single_187_positive))
    print('只有187异常且label为1的在5异常中的占比: ' + str(single_187_positive_187_percent))
    print('只有187异常且label为1的在全部label为1中的占比: ' + str(single_187_positive_total_percent))
    print('只有188异常: ' + str(single_188))
    print('只有188异常且label为1: ' + str(single_188_positive))
    print('只有188异常且label为1的在5异常中的占比: ' + str(single_188_positive_188_percent))
    print('只有188异常且label为1的在全部label为1中的占比: ' + str(single_188_positive_total_percent))
    print('只有197异常: ' + str(single_197))
    print('只有197异常且label为1: ' + str(single_197_positive))
    print('只有197异常且label为1的在5异常中的占比: ' + str(single_197_positive_197_percent))
    print('只有197异常且label为1的在全部label为1中的占比: ' + str(single_197_positive_total_percent))
    print('只有198异常: ' + str(single_198))
    print('只有198异常且label为1: ' + str(single_198_positive))
    print('只有198异常且label为1的在5异常中的占比: ' + str(single_198_positive_198_percent))
    print('只有198异常且label为1的在全部label为1中的占比: ' + str(single_198_positive_total_percent))
    print('全部异常: ' + str(all_abnormal))
    print('全部异常且label为1: ' + str(all_abnormal_positive))
    print('全部异常且label为1的在全部异常中的占比: ' + str(all_abnormal_positive_all_abnormal_percent))
    print('全部异常且label为1的在全部label为1中的占比: ' + str(all_abnormal_positive_total_percent))
    print('任何异常: ' + str(any_abnormal))
    print('任何异常且label为1: ' + str(any_abnormal_positive))
    print('任何异常且label为1的在任何异常中的占比: ' + str(any_abnormal_positive_any_abnormal_percent))
    print('任何异常且label为1的在全部label为1中的占比: ' + str(any_abnormal_positive_total_percent))
    print(data_path + '分析结束!!!!!!!!!!!!!!')
    print('---------------------------------')
    return None


if __name__ == '__main__':
    # 清洗数据
    # bart = DataCleaning()
    # data_list = ['../0326/project/data/round1_testB/disk_sample_smart_log_test_a.csv',
    #              '../0326/project/data/round1_testB/disk_sample_smart_log_test_b.csv',
    #              '../data/disk_sample_smart_log_201806.csv',
    #              '../data/disk_sample_smart_log_201807.csv',
    #              '../data/round2/disk_sample_smart_log_201808.csv', ]
    # for item in data_list:
    #     bart.data_clean(item)

    # 打标签
    # get_label('test_a.csv', '../data/round2/disk_sample_fault_tag_201808.csv', '08')
    # get_label('test_b.csv', '../data/round2/disk_sample_fault_tag_201808.csv', '08')
    # get_label('201808.csv', '../data/round2/disk_sample_fault_tag_201808.csv', '08')
    # get_label('201807.csv', '../data/disk_sample_fault_tag.csv', '07')
    # get_label('201806.csv', '../data/disk_sample_fault_tag.csv', '06')


    # test()

    # 查看5个特征情况
    import glob
    for f in glob.glob('./*'):
        if 'labeled' in f:
            check_abnormal(f)
    # pass
