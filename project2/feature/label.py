# -*- coding:utf-8 -*-
import glob
import pandas as pd
from tqdm import tqdm

class Label:
    def __init__(self):
        self.tag_file = 'data/round2_train/disk_sample_fault_tag.csv'
        self.train_path = 'user_data/tmp_data/*'

    def tag_merge(self):
        print('tag_merge!!!')
        tag = pd.read_csv(self.tag_file)
        tag['tag'] = tag['tag'].astype(str)
        tag['fault_time'] = pd.to_datetime(tag['fault_time'])
        tag = tag.groupby(['serial_number', 'fault_time', 'model'])['tag'].apply(lambda x: '|'.join(x)).reset_index()
        # tag = tag[tag['model'] == 2]
        print('tag: ', tag.shape)
        return tag

    def get_label(self):
        print('get_label!!!')
        tag = self.tag_merge()
        for f in tqdm(glob.glob(self.train_path)):
            if '201806' in f or '201807' in f:
                data = pd.read_csv(f)
                print(f, data.shape)
                data['dt'] = data['dt'].apply(lambda x: ''.join(str(x)[0:4] + '-' + str(x)[4:6] + '-' + str(x)[6:]))
                data['dt'] = pd.to_datetime(data['dt'])
                data = data.merge(tag[['serial_number', 'fault_time', 'model']], how='left',
                                  on=['serial_number', 'model'])
                data['differ_day'] = (data['fault_time'] - data['dt']).dt.days
                data['label'] = 0
                data.loc[(data['differ_day'] <= 30) & (data['differ_day'] >= 0), 'label'] = 1
                data.drop(['fault_time', 'differ_day'], axis=1, inplace=True)
                data.to_csv(f, index=False)
        return None
