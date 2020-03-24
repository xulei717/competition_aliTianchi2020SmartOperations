# -*- coding:utf-8 -*-


import pandas as pd
import datetime


class Feature:
    def feature_extraction(self, df):
        df.drop_duplicates(inplace=True)
        # df["dt"] = df["dt"].apply(lambda x: "".join(str(x)[0:4] + "-" + str(x)[4:6] + "-" + str(x)[6:]))
        df["dt"] = pd.to_datetime(df["dt"])
        df["serial_id"] = df["serial_number"].apply(lambda x: int(x.split("_")[1]))
        df = df.sort_values(by=["serial_id", "dt"], ascending=True)
        df_diff = df[['serial_id', 'smart_1raw', 'smart_5raw', 'smart_7raw', 'smart_199raw', 'dt']].diff(1)
        df_diff['dt'].fillna(datetime.timedelta(days=1), inplace=True)
        df_diff['smart_1raw'] = df_diff['smart_1raw'] / df_diff['dt'].astype('timedelta64[D]').astype(int)
        df_diff['smart_5raw'] = df_diff['smart_5raw'] / df_diff['dt'].astype('timedelta64[D]').astype(int)
        df_diff['smart_7raw'] = df_diff['smart_7raw'] / df_diff['dt'].astype('timedelta64[D]').astype(int)
        df_diff['smart_199raw'] = df_diff['smart_199raw'] / df_diff['dt'].astype('timedelta64[D]').astype(int)
        df_diff.rename(columns=lambda x: x + "_diff", inplace=True)
        df = pd.concat([df, df_diff], axis=1)
        df = df[df.serial_id_diff == 0]
        df.drop(["serial_id", "serial_id_diff", "dt_diff"], axis=1, inplace=True)
        del_cols = ['smart_10raw', 'smart_10_normalized', 'smart_240raw', 'smart_240_normalized',
                    'smart_241raw', 'smart_241_normalized', 'smart_242raw', 'smart_242_normalized',
                    'smart_1raw', 'smart_195raw', 'smart_194_normalized', 'smart_199_normalized']
        df = df[set(df.columns) - set(del_cols)]
        print(df.shape)
        return df
