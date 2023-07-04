#!/usr/bin/env python
# coding: utf-8

import gc

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

pd.set_option('display.min_rows', None)
import time
from pandarallel import pandarallel
from sklearn.metrics import r2_score

pandarallel.initialize()
import warnings

warnings.filterwarnings("ignore")

begin_time = time.time()
print(begin_time)

total_data = pl.read_csv('../gen/train_2.csv')

df_test_data = pl.read_csv('../gen/test_2.csv')

total_data = total_data.to_pandas()
df_test_data = df_test_data.to_pandas()

from sklearn.model_selection import KFold

folds = 5

useless_cols = ['did', 'active_days', 'active_days_preds',
                'watch_nums', 'watch_nums_preds',
                'watch_durations', 'watch_durations_preds']
features = total_data.columns[~total_data.columns.isin(useless_cols)].values
# print (features)
print(len(features))
params = {
    'objective': 'regression',  # 定义的目标函数
    'metric': {'rmse'},
    'boosting_type': 'gbdt',

    # 修改学习率
    'learning_rate': 0.15,
    'max_depth': -1,
    'num_leaves': 2 ** 8,

    'feature_fraction': 0.5,
    'subsample': 0.5,
    'seed': 2023,
    'num_iterations': 3000,
    'nthread': -1,
    'verbose': -1,

    # 'scale_pos_weight':200
}


def train_k_flod(df_train, df_test, label, params, features):
    train_pre = np.zeros((df_train.shape[0], 1))
    test_pre = np.empty((folds, df_test.shape[0], 1))
    print(train_pre.shape, test_pre.shape)
    kf = KFold(folds, shuffle=True, random_state=42)
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    for i, (train_index, valid_index) in enumerate(kf.split(df_train)):
        print(i, )
        #         for_df_train_label = df_train[label]

        tr_x = df_train.iloc[train_index][features]
        tr_y = df_train.iloc[train_index][label].values

        trn_data = lgb.Dataset(tr_x, label=tr_y, silent=False)
        del tr_x, tr_y
        gc.collect()

        vl_x = df_train.iloc[valid_index][features]
        vl_y = df_train.iloc[valid_index][label].values
        val_data = lgb.Dataset(vl_x, label=vl_y, silent=False)

        del vl_y
        gc.collect()

        print('begin')

        model = lgb.train(params,
                          trn_data,
                          valid_sets=[trn_data, val_data],
                          verbose_eval=50,
                          early_stopping_rounds=50)
        del trn_data, val_data
        gc.collect()

        pre = model.predict(vl_x, num_iteration=model.best_iteration).reshape((vl_x.shape[0], 1))
        train_pre[valid_index] = pre
        test_pre[i, :] = model.predict(df_test[features], num_iteration=model.best_iteration).reshape(
            (df_test.shape[0], 1))

        fold_importance_df["importance_{}".format(i)] = model.feature_importance()
        fold_importance_df["importance_gain_{}".format(i)] = model.feature_importance(importance_type='gain')
    fold_importance_df.to_csv("{}_fold_importance_df.csv".format(label, index=None))

    result = r2_score(df_train[label], train_pre)
    print(label, result)
    return test_pre, result


gc.collect()

active_days_, active_days_score = train_k_flod(total_data, df_test_data, 'active_days', params, features)
df_test_data['active_days'] = active_days_.mean(axis=0)
del active_days_
gc.collect()

watch_durations_, watch_durations_score = train_k_flod(total_data, df_test_data, 'watch_durations', params, features)
df_test_data['watch_durations'] = watch_durations_.mean(axis=0)
del watch_durations_
gc.collect()

watch_nums_, watch_nums_score = train_k_flod(total_data, df_test_data, 'watch_nums', params, features)
df_test_data['watch_nums'] = watch_nums_.mean(axis=0)
del watch_nums_
gc.collect()

t = active_days_score * 0.5 + watch_nums_score * 0.25 + watch_durations_score * 0.25

df_test_answer = df_test_data[['did', 'active_days', 'watch_nums', 'watch_durations']]

# active_days 的取值范围 1-7
df_test_answer.loc[df_test_answer['active_days'] >= 7, 'active_days'] = 7
df_test_answer.loc[df_test_answer['active_days'] <= 1, 'active_days'] = 1
df_test_answer.loc[df_test_answer['watch_nums'] <= 0, 'watch_nums'] = 0
df_test_answer.loc[df_test_answer['watch_durations'] <= 0, 'watch_durations'] = 0

df_test_answer.to_csv('../submit/model2.csv', index=None)

print(time.time() - begin_time)
