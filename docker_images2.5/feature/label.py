# -*- coding:utf-8 -*-
import glob
import pandas as pd
from tqdm import tqdm

class Label:
    # def __init__(self, tag_path, train_path):
    #     self.tag_path = tag_path
    #     self.train_path = train_path

    def tag_merge(self, tag_path, month):
        tag = pd.read_csv(tag_path)
        tag['key'] = tag['fault_time'].apply(lambda x: x.split('-')[1])
        tag = tag[tag['key'] == month]
        tag['tag'] = tag['tag'].astype(str)
        tag['fault_time'] = pd.to_datetime(tag['fault_time'])
        tag = tag.groupby(['serial_number', 'fault_time', 'model'])['tag'].apply(lambda x: '|'.join(x)).reset_index()
        return tag

    def get_label(self, tag_path, data_path, month):
        """
        2018tag:manufacturer,model,serial_number,fault_time,tag,key
        """
        tag = self.tag_merge(tag_path, month)
        data = pd.read_csv(data_path)
        data['label'] = 0
        data['dt'] = pd.to_datetime(data['dt'])
        data = data.sort_values(['model', 'serial_number', 'dt'])
        print('data original shape:' + str(data.shape))
        data_last = data.drop_duplicates(['model', 'serial_number'], keep='last')
        print('data last shape:' + str(data_last.shape))
        data_last = data_last.merge(tag, on=['model', 'serial_number'], how='left')
        data_last.loc[~data_last['tag'].isna(), 'label'] = 1
        print('data_last.columns: ', data_last.columns)
        # sys.exit(0)
        data_last.drop(['fault_time', 'tag'], axis=1, inplace=True)
        data = pd.concat([data, data_last])
        data = data.sort_values(['model', 'serial_number', 'dt', 'label'])
        data = data.drop_duplicates(['model', 'serial_number', 'dt'], keep='last')
        print('data done shape:' + str(data.shape))
        data.to_csv(data_path, index=False)
        print('data.columns: ', data.columns)

        return None

