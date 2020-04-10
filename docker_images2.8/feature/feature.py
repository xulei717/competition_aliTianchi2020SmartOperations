# -*- coding:utf-8 -*-


import pandas as pd
import datetime
import re


class Feature:
    # 增加diff列
    def feature_diff(self, df):
        print('feature_diff!!!')
        df.drop_duplicates(['model', 'serial_number', 'dt'], inplace=True)
        # print(df[['model', 'serial_number', 'dt']].head())
        # df["dt"] = df["dt"].apply(lambda x: "".join(str(x)[0:4] + "-" + str(x)[4:6] + "-" + str(x)[6:]))
        df["dt"] = pd.to_datetime(df["dt"])
        # df["serial_id"] = df["serial_number"].apply(lambda x: int(x.split("_")[1]))
        # df["serial_id"] = df[['model', "serial_number"]].apply(lambda x: int(x.values[0]+x.values[1].split("_")[1]))
        df["serial_id"] = df[['model', "serial_number"]].apply(
            lambda x: int(str(x["model"]) + x["serial_number"].split('_')[1]), axis=1)
        df = df.sort_values(by=["serial_id", "dt"], ascending=True)
        # print(df[['model', 'serial_number', 'dt', 'serial_id']].head())
        cols_diff = list(set(['smart_1raw', 'smart_5raw', 'smart_7raw', 'smart_199raw']).intersection(set(df.columns)))
        df_diff = df[['serial_id', 'dt']+cols_diff].diff(1)
        df_diff['dt'].fillna(datetime.timedelta(days=1), inplace=True)
        # df_diff[['serial_id', 'smart_1raw', 'smart_5raw', 'smart_7raw', 'smart_199raw']].fillna(0, inplace=True)  # add
        for col in cols_diff:
            df_diff[col] = df_diff[col] / df_diff['dt'].astype('timedelta64[D]').astype(int)
        # df_diff['smart_5raw'] = df_diff['smart_5raw'] / df_diff['dt'].astype('timedelta64[D]').astype(int)
        # df_diff['smart_7raw'] = df_diff['smart_7raw'] / df_diff['dt'].astype('timedelta64[D]').astype(int)
        # df_diff['smart_199raw'] = df_diff['smart_199raw'] / df_diff['dt'].astype('timedelta64[D]').astype(int)
        df_diff.rename(columns=lambda x: x + "_diff", inplace=True)
        # print('df_diff: ')
        # print(df_diff.head())
        df = pd.concat([df, df_diff], axis=1)
        print('df: ', df.shape)
        df = df[df.serial_id_diff == 0]
        print('df serial_id_diff == 0: ', df.shape)
        df.drop(["serial_id", "serial_id_diff", "dt_diff"], axis=1, inplace=True)
        # del_cols = ['smart_10raw', 'smart_10_normalized', 'smart_240raw', 'smart_240_normalized',
        #             'smart_241raw', 'smart_241_normalized', 'smart_242raw', 'smart_242_normalized',
        #             'smart_1raw', 'smart_195raw', 'smart_194_normalized', 'smart_199_normalized']
        # df = df[set(df.columns) - set(del_cols)]
        print(df.shape)
        return df

    # 删除一些列
    def feature_del(self, df):
        print('feature_del~')
        del_cols = ['smart_10raw', 'smart_10_normalized', 'smart_240raw', 'smart_240_normalized',
                    'smart_241raw', 'smart_241_normalized', 'smart_242raw', 'smart_242_normalized',
                    'smart_1raw', 'smart_195raw', 'smart_194_normalized', 'smart_199_normalized']
        df = df[set(df.columns) - set(del_cols)]
        print(df.shape)
        return df

    # 增加raw/normalized列
    def feature_raw_normalized(self, df):
        print('feature_raw_normalized~')
        raws = [x for x in df.columns if 'raw' in x]
        normalizeds = [x for x in df.columns if 'normalized' in x]
        print('raws: ', len(raws), 'normalizeds: ', len(normalizeds))
        print('raws: ', raws)
        print('normalized: ', normalizeds)
        raws_num = [re.findall(r'\d+', x)[0] for x in raws]
        normalizeds_num = [re.findall(r'\d+', x)[0] for x in normalizeds]
        raws_normalized = list(set(raws_num).intersection(set(normalizeds_num)))
        print('raws_normalized: ', len(raws_normalized), raws_normalized)
        for num in raws_normalized:
            # print(num)
            col = num + '_raw_normalized'
            raw = 'smart_' + num + 'raw'
            normalized = 'smart_' + num + '_normalized'
            df[col] = df[raw] / df[normalized]
            df[col] = df[col].apply(lambda x: None if x == float('inf') or x == float('-inf') else x)
        print(df.shape)
        print(df.columns)
        return df

    # 增加raw/normalized列的diff列
    def feature_raw_normalized_diff(self, df):
        print('feature_raw_normalized_diff!!!')
        df.drop_duplicates(['model', 'serial_number', 'dt'], inplace=True)
        # print(df[['model', 'serial_number', 'dt']].head())
        df["dt"] = pd.to_datetime(df["dt"])
        df["serial_id"] = df[['model', "serial_number"]].apply(
            lambda x: int(str(x["model"]) + x["serial_number"].split('_')[1]), axis=1)
        df = df.sort_values(by=["serial_id", "dt"], ascending=True)
        # print(df[['model', 'serial_number', 'dt', 'serial_id']].head())
        cols_raw_normalized = [x for x in df.columns if 'raw_normalized' in x]
        # df_diff = df[['serial_id', 'smart_1raw', 'smart_5raw', 'smart_7raw', 'smart_199raw', 'dt']].diff(1)
        df_diff = df[['serial_id', 'dt'] + cols_raw_normalized].diff(1)
        df_diff['dt'].fillna(datetime.timedelta(days=1), inplace=True)
        for col in cols_raw_normalized:
            df_diff[col] = df_diff[col] / df_diff['dt'].astype('timedelta64[D]').astype(int)
        df_diff.rename(columns=lambda x: x + "_diff", inplace=True)
        # print('df_diff: ')
        # print(df_diff.head())
        df = pd.concat([df, df_diff], axis=1)
        print('df: ', df.shape)
        df = df[df.serial_id_diff == 0]
        print('df serial_id_diff == 0: ', df.shape)
        df.drop(["serial_id", "serial_id_diff", "dt_diff"], axis=1, inplace=True)
        return df

    # 增加每个列的max,min,mean,skew,nuique,min_max,mean_max,min_mean列
    def feature_addcols(self, df):
        print('feature_addcols')
        print(df.shape)
        df_gb = df.groupby(['model', 'serial_number'])
        for col in df.columns:
            if len(re.findall(r'\d+', col)) != 0:
                # print(col)
                dt = df_gb[col].agg(['max', 'min', 'mean', 'skew', 'nunique'])
                dt.columns = [col + '_' + x for x in dt.columns]
                dt = dt.reset_index()  # model serial_number  ...  smart_1raw_skew  smart_1raw_nunique
                dt[col + '_min_max'] = dt[col + '_max'] - dt[col + '_min']
                dt[col + '_mean_max'] = dt[col + '_max'] - dt[col + '_mean']
                dt[col + '_min_mean'] = dt[col + '_mean'] - dt[col + '_min']
                df = pd.merge(df, dt, on=['model', 'serial_number'], how='left')
                # print(df.shape)
                # print(df.columns)
        print(df.shape)
        print(df.columns)
        return df

    # 删除raw特征列
    def feature_delraw(self, df):
        print('feature_delraw~')
        del_cols = [x for x in df.columns if 'smart' in x and 'raw' in x]
        df = df[set(df.columns) - set(del_cols)]
        print(df.shape)
        return df

    # 删除normalized特征列smart_10_normalized
    def feature_delnormalized(self, df):
        print('feature_delnormalized~')
        del_cols = [x for x in df.columns if 'smart' in x and 'normalized' in x]
        df = df[set(df.columns) - set(del_cols)]
        print(df.shape)
        return df

    # 删除diff特征列
    def feature_deldiff(self, df):
        print('feature_deldiff~')
        del_cols = [x for x in df.columns if 'diff' in x]
        df = df[set(df.columns) - set(del_cols)]
        print(df.shape)
        return df