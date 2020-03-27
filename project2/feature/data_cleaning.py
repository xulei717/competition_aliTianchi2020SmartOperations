# -*- coding:utf-8 -*-
import glob
import pandas as pd
from tqdm import tqdm

class DataCleaning:
    def __init__(self):
        self.train_path = 'data/round2_train/*'
        self.test_path = 'tcdata/disk_sample_smart_log_round2/*'
        self.tmp_path = 'user_data/tmp_data/*'
        keep = [1, 3, 5, 184, 187, 188, 189, 192, 193, 194, 197, 198, 199,
                2, 8, 10, 11, 13, 181, 183, 191, 196, 200, 201, 202, 203, 204, 205, 206, 207,
                220, 221, 224, 225, 227, 228, 250, 254,
                7]
        keep_raw = ['smart_' + str(x) + 'raw' for x in keep]
        keep_normalized = ['smart_' + str(x) + '_normalized' for x in keep]
        self.keeps = keep_raw + keep_normalized + ['dt', "manufacturer", "model", "serial_number"]
        print('keeps: ', len(self.keeps), type(self.keeps))

    # def get_data201808(self):
    #     data = []
    #     for dt in tqdm(glob.glob(self.train_path)):
    #         if 'log_test' in dt:
    #             d = pd.read_csv(dt, index_col=None)
    #             data.append(d)
    #     data = pd.concat(data)
    #     data.to_csv(self.tmp_path + '201808.csv', index=False)

    def dataclean(self):
        print('dataclean!!!')
        for f in tqdm(glob.glob(self.train_path)):
            if '201806' in f or '201807' in f:
                print(f)
                clean_list = list()
                train = pd.read_csv(f, index_col=None, chunksize=100000)
                for chunk in train:
                    clean_list.append(chunk[self.keeps])
                combined = pd.concat(clean_list)
                combined.to_csv(self.tmp_path + f.split('_')[-1], index=False)
                print(f, combined.shape)

    def read_data(self):
        print('read_data!!!')
        #print(glob.glob(self.test_path))
        test_file = glob.glob(self.test_path)
        keep_list = list()
        test = []
        for tf in tqdm(test_file):
            if '201809' in tf:  # docker中需要改成'201809'
                tt = pd.read_csv(tf, index_col=None)
                test.append(tt)
        test = pd.concat(test)
        print('test0: ', test.shape)
        keep_columns = ["manufacturer", "model",
                        'smart_1raw', 'smart_5raw', 'smart_7raw', 'smart_199raw', 'smart_187raw', 'smart_188raw',
                        'smart_5_normalized', 'smart_187_normalized', 'smart_188_normalized', 'smart_197_normalized', 'smart_198_normalized']
        for column in tqdm(test.columns):
            if column in keep_columns or test[column].nunique() > 1:
                keep_list.append(column)
        keep_list = list(set(keep_list).intersection(self.keeps))
        test['dt'] = test['dt'].apply(lambda x: ''.join(str(x)[0:4] + '-' + str(x)[4:6] + '-' + str(x)[6:]))
        test = test[keep_list]
        test.to_csv(self.tmp_path.strip('*') + 'test.csv', index=False)
        print('test: ', test.shape)
        for f in tqdm(glob.glob(self.tmp_path)):
            if '201806' in f or '201807' in f:
                clean_list = list()
                train = pd.read_csv(f, index_col=None, chunksize=100000)
                for chunk in train:
                    clean_list.append(chunk[keep_list+['label']])
                combined = pd.concat(clean_list)
                combined.to_csv(f, index=False)
                print(f, combined.shape)
        return None