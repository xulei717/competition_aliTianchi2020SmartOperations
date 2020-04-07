# -*- coding:utf-8 -*-
import glob
import pandas as pd
from tqdm import tqdm

class Label:
    # def __init__(self, tag_path, train_path):
    #     self.tag_path = tag_path
    #     self.train_path = train_path

    def tag_merge(self, tag_path, year_month):
        tag = pd.read_csv(tag_path)
        if year_month != '201808':
            tag['key'] = tag['fault_time'].apply(lambda x: ''.join(x.split('-')[:2]))
            tag = tag[tag['key'] == year_month]
        tag['tag'] = tag['tag'].astype(str)
        tag['fault_time'] = pd.to_datetime(tag['fault_time'])
        tag = tag.groupby(['serial_number', 'fault_time', 'model'])['tag'].apply(lambda x: '|'.join(x)).reset_index()
        print('tag: ', tag.columns)
        return tag

    def get_label(self, tag_path, data_path, year_month):
        """
        2018tag:manufacturer,model,serial_number,fault_time,tag,key
        """
        print(year_month)
        tag = self.tag_merge(tag_path, year_month)
        data = pd.read_csv(data_path)
        data['label'] = 0
        data['dt'] = pd.to_datetime(data['dt'])
        data = data.sort_values(['model', 'serial_number', 'dt'])
        print('data original shape:' + str(data.shape))
        data_last = data.drop_duplicates(['model', 'serial_number'], keep='last')
        print('data_last shape:' + str(data_last.shape))
        # print('data_last: ', data_last.columns)
        print('tag: ', tag.columns)
        data_last = data_last.merge(tag, on=['model', 'serial_number'], how='left')
        data_last.loc[~data_last['tag'].isna(), 'label'] = 1
        # print('data_last.columns: ', data_last.columns)
        # sys.exit(0)
        data_last.drop(['fault_time', 'tag'], axis=1, inplace=True)
        data = pd.concat([data, data_last])
        data = data.sort_values(['model', 'serial_number', 'dt', 'label'])
        # data = data.drop_duplicates(['model', 'serial_number', 'dt'], keep='last')  # 打标签后数据和初始一样 # 去重不包括dt，可以试一试；把dt换成label可以试一试
        data = data.drop_duplicates(['model', 'serial_number', 'label'], keep='last')  # 打标签后数据大小是负样本每个盘一行数据，正样本每个盘两行数据，label分别为0，1，盘坏的前两天
        print('data done shape:' + str(data.shape))
        data.to_csv(data_path, index=False)
        # print('data.columns: ', data.columns)

        return None

