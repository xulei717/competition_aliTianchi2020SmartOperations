# -*- coding:utf-8 -*-
import glob
import pandas as pd


class DataCleaning:
    def __init__(self):
        self.train_path = '../data/round1_train/*'
        self.test_path = '../data/round1_testB/*'
        self.tmp_path = '../user_data/tmp_data/'

    def read_data(self):
        test_file = glob.glob(self.test_path)[0]
        keep_list = list()
        test = pd.read_csv(test_file, index_col=None)
        for column in test.columns:
            if column in ["manufacturer", "model"] or test[column].nunique() > 1:
                keep_list.append(column)
        test['dt'] = test['dt'].apply(lambda x: ''.join(str(x)[0:4] + '-' + str(x)[4:6] + '-' + str(x)[6:]))
        test[keep_list].to_csv(self.tmp_path + 'test_b.csv', index=False)
        train_file = glob.glob(self.train_path)
        for f in train_file:
            if '201805' in f or '201806' in f:
                clean_list = list()
                train = pd.read_csv(f, index_col=None, chunksize=100000)
                for chunk in train:
                    clean_list.append(chunk[keep_list])
                combined = pd.concat(clean_list)
                combined[combined['model'] == 2].to_csv(self.tmp_path + f.split('_')[-1], index=False)
        return None