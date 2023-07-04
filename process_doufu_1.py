#!/usr/bin/env python
# coding: utf-8


import gc

import pandas as pd
import polars as pl

pd.set_option('display.min_rows', None)
import datetime
import time
from pandarallel import pandarallel

pandarallel.initialize()
import warnings

warnings.filterwarnings("ignore")

begin_time = time.time()
print(begin_time)


def sliding_window(df, end_date, day=7):
    start_date = end_date - datetime.timedelta(day)

    df = df.filter(
        pl.col("date_day").is_between(start_date, end_date),
    )
    print(df['date'].max(), df['date'].min())
    return df


def get_label(df_behaviors, end_date, day=7):
    user_behaviors_label_windows = sliding_window(df_behaviors,
                                                  end_date,
                                                  day=day)
    user_history_hebaviors = sliding_window(df_behaviors,
                                            end_date - datetime.timedelta(7),
                                            day=120)
    out = user_behaviors_label_windows.groupby("did").agg(
        [
            pl.col("date_day").n_unique().alias('active_days'),
            pl.col("vid").count().alias('watch_nums'),
            pl.col("vts").sum().alias('watch_durations'),
        ]
    )
    return out, user_history_hebaviors


vid_info = pl.read_csv('../data/vid_info.csv')

vid_info = vid_info.with_columns(pl.col('online_time').str.strptime(pl.Datetime))

user_behaviors = pl.read_csv('../data/user_behaviors.csv')

user_behaviors = user_behaviors.with_columns(
    [
        (pl.col("timestamp") * 1000).cast(pl.Datetime(time_unit="ms")).alias("date")
    ]
)

user_behaviors = user_behaviors.with_columns(
    [
        pl.col("date").dt.replace_time_zone("UTC")
    ]
)

user_behaviors = user_behaviors.with_columns(
    [
        pl.col("date").dt.convert_time_zone("Asia/Shanghai")
    ]
)

user_behaviors = user_behaviors.with_columns(
    [
        pl.col("date").dt.date().alias('date_day'),
    ]
)

user_behaviors = user_behaviors.join(vid_info[['vid', 'online_time']], on=['vid'], how='left')
user_behaviors = user_behaviors.with_columns([
    pl.col('online_time').dt.epoch(time_unit="us").alias('data_diff') / 1000000
])
user_behaviors = user_behaviors.with_columns([
    (pl.col('timestamp') - pl.col('data_diff')).alias('u_date') / (3600 * 24)
])
user_behaviors = user_behaviors.drop('online_time')
user_behaviors = user_behaviors.drop('data_diff')

vid_info = vid_info.drop('stars')
vid_info = vid_info.drop('tags')
vid_info = vid_info.drop('img_url')


def make_number_feats(df, vid_info, days):
    end_date = df['date'].max()
    # 截取需要的时间范围的历史数据
    df_temp = sliding_window(df, end_date, days)
    df_temp = df_temp.join(vid_info, on=["vid"], how="left")
    del vid_info
    gc.collect()
    # vts / duration , 倒序权重 ， 指数权重
    df_temp = df_temp.with_columns(
        [
            (pl.col('vts') / pl.col('duration')).alias("vts/duration"),
            #             ((end_date - pl.col('date_day')).dt.days() + 1).alias("weight"),
            ((end_date - pl.col('date_day')).dt.days() + 1).apply(lambda x: 0.95 ** x).alias("weight_95"),
        ]
    )

    # 统计其他id类字段出现的频次
    add_feat = []
    for c_col in ['vid']:
        add_feat.append('{}_active_days'.format(c_col))
        add_feat.append('{}_watch_nums'.format(c_col))
        add_feat.append('{}_watch_durations'.format(c_col))
        t = df_temp.groupby([c_col]).agg(
            [
                pl.col("date_day").n_unique().alias('{}_active_days'.format(c_col)),
                pl.col("vid").count().alias('{}_watch_nums'.format(c_col)),
                pl.col("vts").filter(pl.col("vts").is_not_null()).sum().alias('{}_watch_durations'.format(c_col)),
            ]
        )
        df_temp = df_temp.join(t, on=[c_col], how='left')
        del t
        gc.collect()

    # 结合did字段 统计其他id类字段出现的频次
    for c_col in ['cid']:
        add_feat.append('did_{}_active_days'.format(c_col))
        add_feat.append('did_{}_watch_nums'.format(c_col))
        add_feat.append('did_{}_watch_durations'.format(c_col))
        t = df_temp.groupby(['did', c_col]).agg(
            [
                pl.col("date_day").n_unique().alias('did_{}_active_days'.format(c_col)),
                pl.col("vid").count().alias('did_{}_watch_nums'.format(c_col)),
                pl.col("vts").filter(pl.col("vts").is_not_null()).sum().alias('did_{}_watch_durations'.format(c_col)),
            ]
        )
        df_temp = df_temp.join(t, on=['did', c_col], how='left')
        del t
        gc.collect()

    # 结合is_intact字段 统计其他id类字段出现的频次
    f_intact = []
    if days <= 7:
        f_intact = ['cid']
    else:
        f_intact = ['cid']
    for c_col in f_intact:
        add_feat.append('is_intact_{}_active_days'.format(c_col))
        add_feat.append('is_intact_{}_watch_nums'.format(c_col))
        add_feat.append('is_intact_{}_watch_durations'.format(c_col))
        t = df_temp.groupby(['is_intact', c_col]).agg(
            [
                pl.col("date_day").n_unique().alias('is_intact_{}_active_days'.format(c_col)),
                pl.col("vid").count().alias('is_intact_{}_watch_nums'.format(c_col)),
                pl.col("vts").filter(pl.col("vts").is_not_null()).sum().alias(
                    'is_intact_{}_watch_durations'.format(c_col)),
            ]
        )
        df_temp = df_temp.join(t, on=['is_intact', c_col], how='left')
        del t
        gc.collect()

    if days == 1:
        df_feats = df_temp.groupby("did").agg(
            [
                pl.col("vts").filter(pl.col("is_intact") == 1).sum().alias("is_intact_1_vts_{}".format(days)),
                pl.col("vid").filter(pl.col("is_intact") == 1).count().alias("is_intact_1_vid_{}".format(days)),

                pl.col("vts").filter(pl.col("is_intact") == 2).sum().alias("is_intact_2_vts_{}".format(days)),
                pl.col("vid").filter(pl.col("is_intact") == 2).count().alias("is_intact_2_vid_{}".format(days)),

                pl.col("vts").filter(pl.col("is_intact") == 3).sum().alias("is_intact_3_vts_{}".format(days)),
                pl.col("vid").filter(pl.col("is_intact") == 3).count().alias("is_intact_3_vid_{}".format(days)),

                pl.col("vts").filter(pl.col("is_intact") == 4).sum().alias("is_intact_4_vts_{}".format(days)),
                pl.col("vid").filter(pl.col("is_intact") == 4).count().alias("is_intact_4_vid_{}".format(days)),

                #                 pl.col("vts").filter(pl.col("is_intact") == 5).sum().alias("is_intact_5_vts_{}".format(days)),
                #                 pl.col("vid").filter(pl.col("is_intact") == 5).count().alias("is_intact_5_vid_{}".format(days)),

                pl.col("vts").filter(pl.col("is_intact") == 7).sum().alias("is_intact_7_vts_{}".format(days)),
                pl.col("vid").filter(pl.col("is_intact") == 7).count().alias("is_intact_7_vid_{}".format(days)),

                #                 pl.col("vts").filter(pl.col("is_intact") == 9).sum().alias("is_intact_9_vts_{}".format(days)),
                #                 pl.col("vid").filter(pl.col("is_intact") == 9).count().alias("is_intact_9_vid_{}".format(days)),

                pl.col("vid").count().alias('did_vid_{}_count'.format(days)),

                pl.col("vid").n_unique().alias('did_vid_{}_nunique'.format(days)),

                (pl.col("vid").n_unique() / pl.col("vid").count()).alias('did_vid_{}_nunique_count'.format(days)),

                ((pl.col("timestamp").max() - pl.col("timestamp").min()) / 3600).alias('ts_ptp_{}'.format(days)),

                pl.col("cid").n_unique().alias('did_cid_{}_nunique'.format(days)),

                pl.col("classify_id").n_unique().alias('did_classify_id_{}_nunique'.format(days)),

                pl.col("series_id").n_unique().alias('did_series_id_{}_nunique'.format(days)),
            ]
        )
    else:
        df_feats = df_temp.groupby("did").agg(
            [
                pl.col("vts").filter(pl.col("is_intact") == 1).sum().alias("is_intact_1_vts_{}".format(days)),
                pl.col("vid").filter(pl.col("is_intact") == 1).count().alias("is_intact_1_vid_{}".format(days)),

                pl.col("vts").filter(pl.col("is_intact") == 2).sum().alias("is_intact_2_vts_{}".format(days)),
                pl.col("vid").filter(pl.col("is_intact") == 2).count().alias("is_intact_2_vid_{}".format(days)),

                pl.col("vts").filter(pl.col("is_intact") == 3).sum().alias("is_intact_3_vts_{}".format(days)),
                pl.col("vid").filter(pl.col("is_intact") == 3).count().alias("is_intact_3_vid_{}".format(days)),

                pl.col("vts").filter(pl.col("is_intact") == 4).sum().alias("is_intact_4_vts_{}".format(days)),
                pl.col("vid").filter(pl.col("is_intact") == 4).count().alias("is_intact_4_vid_{}".format(days)),

                #                 pl.col("vts").filter(pl.col("is_intact") == 5).sum().alias("is_intact_5_vts_{}".format(days)),
                #                 pl.col("vid").filter(pl.col("is_intact") == 5).count().alias("is_intact_5_vid_{}".format(days)),

                pl.col("vts").filter(pl.col("is_intact") == 7).sum().alias("is_intact_7_vts_{}".format(days)),
                pl.col("vid").filter(pl.col("is_intact") == 7).count().alias("is_intact_7_vid_{}".format(days)),

                pl.col("vts").filter(pl.col("is_intact") == 9).sum().alias("is_intact_9_vts_{}".format(days)),
                pl.col("vid").filter(pl.col("is_intact") == 9).count().alias("is_intact_9_vid_{}".format(days)),

                pl.col("vid").count().alias('did_vid_{}_count'.format(days)),

                pl.col("vid").n_unique().alias('did_vid_{}_nunique'.format(days)),

                (pl.col("vid").n_unique() / pl.col("vid").count()).alias('did_vid_{}_nunique_count'.format(days)),

                ((pl.col("timestamp").max() - pl.col("timestamp").min()) / 3600).alias('ts_ptp_{}'.format(days)),

                pl.col("date_day").n_unique().alias('did_date_{}_nunique'.format(days)),

                (pl.col("date_day").n_unique() / days).alias('did_date/days_{}_nunique'.format(days)),

                pl.col("cid").n_unique().alias('did_cid_{}_nunique'.format(days)),

                pl.col("classify_id").n_unique().alias('did_classify_id_{}_nunique'.format(days)),

                pl.col("series_id").n_unique().alias('did_series_id_{}_nunique'.format(days)),
            ]
        )

    key = 'vts'
    a = df_temp.groupby("did").agg(
        [
            (pl.col("{}".format(key)) * (pl.col('weight_95'))).mean().alias('w95_did_{}_{}_mean'.format(key, days)),
            #             (pl.col("{}".format(key)) * (1/ pl.col('weight'))).mean().alias('w_did_{}_{}_mean'.format(key,days)),
            pl.col("{}".format(key)).filter(pl.col("{}".format(key)).is_not_null()).mean().alias(
                'did_{}_{}_mean'.format(key, days)),
            pl.col("{}".format(key)).filter(pl.col("{}".format(key)).is_not_null()).std().alias(
                'did_{}_{}_std'.format(key, days)),
            pl.col("{}".format(key)).min().alias('did_{}_{}_min'.format(key, days)),
            pl.col("{}".format(key)).max().alias('did_{}_{}_max'.format(key, days)),
            pl.col("{}".format(key)).filter(pl.col("{}".format(key)).is_not_null()).sum().alias(
                'did_{}_{}_sum'.format(key, days)),
        ])

    df_feats = df_feats.join(a, on=["did"], how="left")
    del a
    gc.collect()

    #     u_date
    key = 'u_date'
    a = df_temp.groupby("did").agg(
        [
            (pl.col("{}".format(key)) * (pl.col('weight_95'))).mean().alias('w95_did_{}_{}_mean'.format(key, days)),
            #             (pl.col("{}".format(key)) * (1/ pl.col('weight'))).mean().alias('w_did_{}_{}_mean'.format(key,days)),
            pl.col("{}".format(key)).filter(pl.col("{}".format(key)).is_not_null()).mean().alias(
                'did_{}_{}_mean'.format(key, days)),
            pl.col("{}".format(key)).filter(pl.col("{}".format(key)).is_not_null()).std().alias(
                'did_{}_{}_std'.format(key, days)),
            pl.col("{}".format(key)).min().alias('did_{}_{}_min'.format(key, days)),
            pl.col("{}".format(key)).max().alias('did_{}_{}_max'.format(key, days)),
            pl.col("{}".format(key)).filter(pl.col("{}".format(key)).is_not_null()).sum().alias(
                'did_{}_{}_sum'.format(key, days)),
        ])

    df_feats = df_feats.join(a, on=["did"], how="left")
    del a
    gc.collect()

    add_feat_use = []
    if days == 1:
        for f in add_feat:
            if f not in [
                'is_intact_cid_active_days',
                'did_cid_active_days',
                'is_intact_series_id_active_days',
                'vid_active_days',
                'cid_active_days'
            ]:
                add_feat_use.append(f)
    else:
        add_feat_use = add_feat

    for key in [
                   'vts/duration',
               ] + add_feat_use:
        a = df_temp.groupby("did").agg(
            [
                (pl.col("{}".format(key)) * (pl.col('weight_95'))).mean().alias('w95_did_{}_{}_mean'.format(key, days)),
                #            (pl.col("{}".format(key)) * (1/ pl.col('weight'))).mean().alias('w_did_{}_{}_mean'.format(key,days)),
                pl.col("{}".format(key)).filter(pl.col("{}".format(key)).is_not_null()).mean().alias(
                    'did_{}_{}_mean'.format(key, days)),
                pl.col("{}".format(key)).filter(pl.col("{}".format(key)).is_not_null()).std().alias(
                    'did_{}_{}_std'.format(key, days)),
                pl.col("{}".format(key)).min().alias('did_{}_{}_min'.format(key, days)),
                pl.col("{}".format(key)).max().alias('did_{}_{}_max'.format(key, days)),
            ]
        )
        df_feats = df_feats.join(a, on=["did"], how="left")
        del a
        gc.collect()

    del df_temp
    gc.collect()

    return df_feats


def create_sample(user_history_behaviors, df_label):
    # 训练集特征窗口
    for days in [1, 3, 5, 7, 15, 30, 60, 120]:
        print(days)
        df_feats = make_number_feats(user_history_behaviors, vid_info, days=days)
        df_label = df_label.join(df_feats, on='did', how='left')
        del df_feats
    return df_label


of_day = 7

# 训练数据信息1
df_train_label_1, user_train_history_behaviors_1 = get_label(user_behaviors,
                                                             user_behaviors['date_day'].max() - datetime.timedelta(3),
                                                             day=6)

df_train_data_1 = create_sample(user_train_history_behaviors_1, df_train_label_1)
del user_train_history_behaviors_1, df_train_label_1
gc.collect()

# 训练数据信息1
df_train_label_2, user_train_history_behaviors_2 = get_label(user_behaviors,
                                                             user_behaviors['date_day'].max() - datetime.timedelta(7),
                                                             day=6)

df_train_data_2 = create_sample(user_train_history_behaviors_2, df_train_label_2)
del user_train_history_behaviors_2, df_train_label_2
gc.collect()

# 训练数据信息1
df_train_label_3, user_train_history_behaviors_3 = get_label(user_behaviors,
                                                             user_behaviors['date_day'].max() - datetime.timedelta(14),
                                                             day=6)

df_train_data_3 = create_sample(user_train_history_behaviors_3, df_train_label_3)
del user_train_history_behaviors_3, df_train_label_3
gc.collect()

# 生成验证集label
df_valid_label, user_valid_history_behaviors = get_label(user_behaviors, user_behaviors['date_day'].max(), day=6)

# 构建验证集样本
df_valid_data = create_sample(user_valid_history_behaviors, df_valid_label)
del user_valid_history_behaviors, df_valid_label
gc.collect()

total_data = pd.concat([
    df_train_data_3.to_pandas(),
    df_train_data_2.to_pandas(),
    df_train_data_1.to_pandas(),
    df_valid_data.to_pandas()
])

del df_valid_data, df_train_data_1, df_train_data_2, df_train_data_3
gc.collect()

# 测试集特征
df_test_data = create_sample(user_behaviors, user_behaviors.unique(subset=['did'], keep='first')[['did']])
df_test_data = df_test_data.to_pandas()

del user_behaviors
gc.collect()

# from sklearn.model_selection import KFold
# folds = 5


useless_cols = ['did', 'active_days', 'active_days_preds',
                'watch_nums', 'watch_nums_preds',
                'watch_durations', 'watch_durations_preds']

total_data.to_csv('../gen/train_1.csv', index=False)
df_test_data.to_csv('../gen/test_1.csv', index=False)

print(time.time() - begin_time)
