# -*- coding:utf-8 -*-


import pandas as pd
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
pd.options.display.width = 500
import numpy
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import lightgbm


class BasicModel:
    # lightgbm模型
    def lightgbm_model(self, train, test, features, target):
        fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)

        X = train[features].copy()
        y = train[target]
        pred = numpy.zeros(len(test))
        oof = numpy.zeros(len(X))

        for index, (train_idx, val_idx) in enumerate(fold.split(X, y)):
            train_X = X.iloc[train_idx]
            train_y = y.iloc[train_idx]
            val_X = X.iloc[val_idx]
            val_y = y.iloc[val_idx]
            model = lightgbm.LGBMClassifier(
                boosting_type="gbdt",
                #metric='auc',
                learning_rate=0.001,
                n_estimators=5000,
                num_leaves=100,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=2020)
            model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=500)
            val_pred = model.predict(val_X)
            oof[val_idx] = val_pred
            val_f1 = metrics.f1_score(val_y, val_pred)
            print(index, 'val f1', val_f1)

            test_pred = model.predict(test[features])
            pred += test_pred / 5

        print('oof f1', metrics.f1_score(oof, y))
        result = test[["manufacturer", "model", "serial_number", "dt", "smart_5raw", "smart_187raw", "smart_188raw"]]
        # pred = [1 if p > 0.5 else 0 for p in pred]
        result['pred'] = pred
        print(result.shape)
        result = result[result.pred == 1]
        result = result[(result.smart_5raw > 0)|((result.smart_187raw > 0)|(result.smart_188raw > 0))]
        print(result.shape)
        result = result.groupby(["manufacturer", "model", "serial_number"])["dt"].max().reset_index()
        return result
