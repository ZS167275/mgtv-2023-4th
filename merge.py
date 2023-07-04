#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import polars as pl

data1 = pl.read_csv('../submit/model1.csv')

data2 = pl.read_csv('../submit/model1.csv')

data1 = data1.to_pandas()

data2 = data2.to_pandas()

data1 = data1.sort_values('did').reset_index(drop=True)

data2 = data2.sort_values('did').reset_index(drop=True)

a = pd.concat([data1['did'], data2['did']], axis=1)
a.columns = ['did1', 'did2']

data1.columns = ['did', 'active_days_1', 'watch_nums_1', 'watch_durations_1']

data2.columns = ['did', 'active_days_2', 'watch_nums_2', 'watch_durations_2']

data = pd.merge(data1, data2, on=['did'], how='left')

data.head()

data['active_days'] = 0.5 * data['active_days_1'] + 0.5 * data['active_days_2']

data['watch_nums'] = 0.5 * data['watch_nums_1'] + 0.5 * data['watch_nums_2']

data['watch_durations'] = 0.5 * data['watch_durations_1'] + 0.5 * data['watch_durations_2']

# active_days 的取值范围 1-7
data.loc[data['active_days'] >= 7, 'active_days'] = 7
data.loc[data['active_days'] <= 1, 'active_days'] = 1

data[['did', 'active_days', 'watch_nums', 'watch_durations']].to_csv('../submit/mg3.csv', index=False)
