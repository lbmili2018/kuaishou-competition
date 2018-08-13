## -*- coding: utf-8 -*-

'''480特征提取'''
import pandas as pd
import numpy as np
import gc 
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

train_path = '/home/kesci/train_480.csv'
train_path1 = '/home/kesci/train1_480.csv'
train_path2 = '/home/kesci/train2_480.csv'
train_path3 = '/home/kesci/train3_480.csv'
test_path = '/home/kesci/test_480.csv'

dataset_1_feat_dir = '/home/kesci/dataset_1_feat'
dataset_1_label_dir = '/home/kesci/dataset_1_label'
dataset_2_feat_dir = '/home/kesci/dataset_2_feat'
dataset_2_label_dir = '/home/kesci/dataset_2_label'
dataset_3_feat_dir = '/home/kesci/dataset_3_feat'
dataset_3_label_dir = '/home/kesci/dataset_3_label'
dataset_4_feat_dir = '/home/kesci/dataset_4_feat'

def get_train_label(feat_path,label_path):
    #获取训练集用户的id
    feat_register = pd.read_csv(feat_path + '/register.csv', usecols=['user_id'])
    feat_data_id = np.unique(feat_register)
    
    #出现在其中一个表中的用户即为活跃用户
    label_launch = pd.read_csv(label_path + '/launch.csv', usecols=['user_id'])
    label_video = pd.read_csv(label_path + '/video.csv', usecols=['user_id'])
    label_activity = pd.read_csv(label_path + '/activity.csv', usecols=['user_id'])
    label_data_id = np.unique(pd.concat([label_launch,label_video,label_activity]))

    train_label = []
    for i in feat_data_id:
        if i in label_data_id:
            train_label.append(1)
        else:
            train_label.append(0)
    train_data = pd.DataFrame()
    train_data['user_id'] = feat_data_id
    train_data['label'] = train_label
    return train_data

def get_test_id(feat_path):
    feat_register = pd.read_csv(feat_path + '/register.csv', usecols=['user_id'])
    test_data = pd.DataFrame()
    feat_data_id = np.unique(feat_register)
    test_data['user_id'] = feat_data_id
    return test_data


####################################构建特征############################################

def get_register_feature(register):
    feature = pd.DataFrame()
    feature['user_id'] = register['user_id'].drop_duplicates()
    begin_day = np.min(register['register_day'])
    end_day = np.max(register['register_day'])
    feature['register_day'] = register['register_day']
    feature['register_day_sub_begin_day'] = register['register_day']-begin_day
    feature['register_type'] = register['register_type']
    feature['device_type'] = register['device_type']
    
    # 对类别值统计
    feature['device_type_count'] = (register.groupby(by=["device_type"])["device_type"].transform("count")).astype(np.uint16)
    len_device = len(register)*1.0
    feature['device_type_count_rate'] = feature['device_type_count'] / len_device
    
    # 类型值编码
    register_type_dummies =pd.get_dummies(register['register_type'],prefix='register_type')
    feature = feature.join(register_type_dummies)
    
    # device_type_count_type_dummies =pd.get_dummies(registerCopy['device_type_count'],prefix='device_type_count_type')
    # feature = feature.join(device_type_count_type_dummies)
    
    feature['register_rest_day'] = register['register_day']-end_day
    feature['register_rest_day'] = -feature['register_rest_day']+1
    feature['divisor_day'] = feature['register_rest_day']
    feature.loc[feature['divisor_day']>16,'divisor_day']=16
    return feature

def launch_slide_window(launch):
    launch_total_count = launch[['user_id']].groupby(['user_id']).size().rename('launch_total_count').reset_index()
    
    # 加入权重信息
    launch['app_launch_log_flag'] = launch['app_launch_log_flag'].fillna(0)
    launch_date = np.max(launch['launch_day'])
    launch['weight'] = launch_date + 1- launch['launch_day']
    launch['weight'] = 1 / launch['weight']
    launch['app_launch_log_flag'] = launch['app_launch_log_flag'] * launch['weight']
    launch_time_feat = launch.groupby(['user_id'])['app_launch_log_flag'].sum().rename('windows_launch_feat').reset_index()

    return launch_total_count,launch_time_feat

def get_launch_feature(launch):
    feature = pd.DataFrame()
    feature['user_id'] = launch['user_id'].drop_duplicates()
    end_day = np.max(launch['launch_day'])
    #滑窗 窗口内launch次数
    launch_total_count_all, launch_time_feat_all=launch_slide_window(launch)
    launch_total_count_1 , launch_time_feat_1=launch_slide_window(launch[(launch['launch_day']>=end_day) & (launch['launch_day']<=end_day)])
    launch_total_count_2, launch_time_feat_2=launch_slide_window(launch[(launch['launch_day']>=end_day-1) & (launch['launch_day']<=end_day)])
    launch_total_count_3, launch_time_feat_3=launch_slide_window(launch[(launch['launch_day']>=end_day-2) & (launch['launch_day']<=end_day)])
    launch_total_count_5, launch_time_feat_5=launch_slide_window(launch[(launch['launch_day']>=end_day-4) & (launch['launch_day']<=end_day)])
    launch_total_count_7, launch_time_feat_7=launch_slide_window(launch[(launch['launch_day']>=end_day-6) & (launch['launch_day']<=end_day)])
    launch_total_count_10, launch_time_feat_10=launch_slide_window(launch[(launch['launch_day']>=end_day-9) & (launch['launch_day']<=end_day)])
    #launch相隔天数特征
    launch_day_diff = pd.concat([launch['user_id'],launch.groupby(['user_id']).diff().rename({'launch_day':'launch_day_diff'},axis=1)],axis=1)
    launch_day_diff_max = launch_day_diff.groupby(['user_id'])['launch_day_diff'].max().rename('launch_day_diff_max').reset_index()
    launch_day_diff_min = launch_day_diff.groupby(['user_id'])['launch_day_diff'].min().rename('launch_day_diff_min').reset_index()
    launch_day_diff_mean = launch_day_diff.groupby(['user_id'])['launch_day_diff'].mean().rename('launch_day_diff_mean').reset_index()
    launch_day_diff_std = launch_day_diff.groupby(['user_id'])['launch_day_diff'].std().rename('launch_day_diff_std').reset_index()
    launch_day_diff_kurt = launch_day_diff.groupby(['user_id'])['launch_day_diff'].agg(lambda x: pd.Series.kurt(x)).rename('launch_day_diff_kurt').reset_index()
    launch_day_diff_skew = launch_day_diff.groupby(['user_id'])['launch_day_diff'].skew().rename('launch_day_diff_skew').reset_index()
    launch_day_diff_last = launch_day_diff.groupby(['user_id'])['launch_day_diff'].last().rename('launch_day_diff_last').reset_index()

    #最后一次launch在哪一天
    launch_last_day = launch.groupby(['user_id'])['launch_day'].max().rename('launch_last_day').reset_index()
    
    #第一次启动是在哪一天
    launch_first_day = launch.groupby(['user_id'])['launch_day'].min().rename('launch_first_day').reset_index()

    #最后一次launch到end day隔了多久
    launch_rest_day = (launch.groupby(['user_id'])['launch_day'].max()-end_day).rename('launch_rest_day').reset_index()
    launch_rest_day['launch_rest_day'] = -launch_rest_day['launch_rest_day']
    #按天统计特征
    launch_count = launch.groupby(['user_id','launch_day']).agg({'launch_day':'count'}).rename({'launch_day':'launch_count'},axis=1).reset_index()
    launch_count_max = launch_count.groupby(['user_id'])['launch_count'].max().rename('launch_count_max').reset_index()
    launch_count_min = launch_count.groupby(['user_id'])['launch_count'].min().rename('launch_count_min').reset_index()
    launch_count_mean = launch_count.groupby(['user_id'])['launch_count'].mean().rename('launch_count_mean').reset_index()
    launch_count_std = launch_count.groupby(['user_id'])['launch_count'].std().rename('launch_count_std').reset_index()
    launch_count_kurt = launch_count.groupby(['user_id'])['launch_count'].agg(lambda x: pd.Series.kurt(x)).rename('launch_count_kurt').reset_index()
    launch_count_skew = launch_count.groupby(['user_id'])['launch_count'].skew().rename('launch_count_skew').reset_index()
    launch_count_last = launch_count.groupby(['user_id'])['launch_count'].last().rename('launch_count_last').reset_index()
    launch_day_mode = launch_count.groupby(['user_id'])['launch_count'].agg(lambda x: np.mean(pd.Series.mode(x))).rename('launch_day_mode').reset_index()
    #最多launch的是哪一天
    most_launch_day = launch_count.groupby('user_id').apply(lambda x: x[x.launch_count==x.launch_count.max()]).rename({'launch_day':'most_launch_day'},axis=1).drop('launch_count',axis=1).groupby('user_id')['most_launch_day'].max().reset_index()
    most_launch_day_sub_end_day = (most_launch_day-end_day).rename({'most_launch_day':'most_launch_day_sub_end_day'},axis=1)
    #MERGE
    
    #滑窗特征
    feature = pd.merge(feature,launch_total_count_all,how='left',on='user_id')
    feature = feature.rename({'launch_total_count':'launch_total_count_all'},axis=1)
    feature = pd.merge(feature,launch_total_count_1,how='left',on='user_id')
    feature = feature.rename({'launch_total_count':'launch_total_count_1'},axis=1)
    feature = pd.merge(feature,launch_total_count_2,how='left',on='user_id')
    feature = feature.rename({'launch_total_count':'launch_total_count_2'},axis=1)
    feature = pd.merge(feature,launch_total_count_3,how='left',on='user_id')
    feature = feature.rename({'launch_total_count':'launch_total_count_3'},axis=1)
    feature = pd.merge(feature,launch_total_count_5,how='left',on='user_id')
    feature = feature.rename({'launch_total_count':'launch_total_count_5'},axis=1)
    feature = pd.merge(feature,launch_total_count_7,how='left',on='user_id')
    feature = feature.rename({'launch_total_count':'launch_total_count_7'},axis=1)
    feature = pd.merge(feature,launch_total_count_10,how='left',on='user_id')
    feature = feature.rename({'launch_total_count':'launch_total_count_10'},axis=1)
    # 加入时间权重
    feature = pd.merge(feature,launch_time_feat_all,how='left',on='user_id')
    feature = feature.rename({'launch_time_feat':'launch_time_feat_all'},axis=1)
    feature = pd.merge(feature,launch_time_feat_1,how='left',on='user_id')
    feature = feature.rename({'launch_time_feat':'launch_time_feat_1'},axis=1)
    feature = pd.merge(feature,launch_time_feat_2,how='left',on='user_id')
    feature = feature.rename({'launch_time_feat':'launch_time_feat_2'},axis=1)
    feature = pd.merge(feature,launch_time_feat_3,how='left',on='user_id')
    feature = feature.rename({'launch_time_feat':'launch_time_feat_3'},axis=1)
    feature = pd.merge(feature,launch_time_feat_5,how='left',on='user_id')
    feature = feature.rename({'launch_time_feat':'launch_time_feat_5'},axis=1)
    feature = pd.merge(feature,launch_time_feat_7,how='left',on='user_id')
    feature = feature.rename({'launch_time_feat':'launch_time_feat_7'},axis=1)
    feature = pd.merge(feature,launch_time_feat_10,how='left',on='user_id')
    feature = feature.rename({'launch_time_feat':'launch_time_feat_10'},axis=1)
    
    feature = pd.merge(feature,launch_day_diff_max,how='left',on='user_id')
    feature = pd.merge(feature,launch_day_diff_min,how='left',on='user_id')
    feature = pd.merge(feature,launch_day_diff_mean,how='left',on='user_id')
    feature = pd.merge(feature,launch_day_diff_std,how='left',on='user_id')
    feature = pd.merge(feature,launch_day_diff_kurt,how='left',on='user_id')
    feature = pd.merge(feature,launch_day_diff_skew,how='left',on='user_id')
    feature = pd.merge(feature,launch_day_diff_last,how='left',on='user_id')
    feature = pd.merge(feature,launch_last_day,how='left',on='user_id')
    feature = pd.merge(feature,launch_first_day,how='left',on='user_id')
    feature = pd.merge(feature,launch_rest_day,how='left',on='user_id')
    feature = pd.merge(feature,launch_count_max,how='left',on='user_id')
    feature = pd.merge(feature,launch_count_min,how='left',on='user_id')
    feature = pd.merge(feature,launch_count_mean,how='left',on='user_id')
    feature = pd.merge(feature,launch_count_std,how='left',on='user_id')
    feature = pd.merge(feature,launch_count_kurt,how='left',on='user_id')
    feature = pd.merge(feature,launch_count_skew,how='left',on='user_id')
    feature = pd.merge(feature,launch_count_last,how='left',on='user_id')
    feature = pd.merge(feature,launch_day_mode,how='left',on='user_id')
    feature = pd.merge(feature,most_launch_day,how='left',on='user_id')
    feature = pd.merge(feature,most_launch_day_sub_end_day,how='left',on='user_id')
    return feature

def video_slide_window(video):
    video_total_count = video[['user_id']].groupby(['user_id']).size().rename('video_total_count').reset_index()
     # 加入权重信息
    video['app_video_log_flag'] = video['app_video_log_flag'].fillna(0)
    video_date = np.max(video['video_day'])
    video['weight'] = video_date + 1- video['video_day']
    video['weight'] = 1 / video['weight']
    video['app_video_log_flag'] = video['app_video_log_flag'] * video['weight']
    video_time_feat = video.groupby(['user_id'])['app_video_log_flag'].sum().rename('windows_video_feat').reset_index()
    return video_total_count,video_time_feat

def get_video_feature(video):
    feature = pd.DataFrame()
    feature['user_id'] = video['user_id'].drop_duplicates()
    end_day = np.max(video['video_day'])
    #滑窗 窗口内video次数
    video_total_count_all,video_time_feat_all=video_slide_window(video)
    video_total_count_1,video_time_feat_1=video_slide_window(video[(video['video_day']>=end_day) & (video['video_day']<=end_day)])
    video_total_count_2,video_time_feat_2=video_slide_window(video[(video['video_day']>=end_day-1) & (video['video_day']<=end_day)])
    video_total_count_3,video_time_feat_3=video_slide_window(video[(video['video_day']>=end_day-2) & (video['video_day']<=end_day)])
    video_total_count_5,video_time_feat_5=video_slide_window(video[(video['video_day']>=end_day-4) & (video['video_day']<=end_day)])
    video_total_count_7,video_time_feat_7=video_slide_window(video[(video['video_day']>=end_day-6) & (video['video_day']<=end_day)])
    video_total_count_10,video_time_feat_10=video_slide_window(video[(video['video_day']>=end_day-9) & (video['video_day']<=end_day)])
    #video相隔天数特征
    video_day_diff = pd.concat([video['user_id'],video.groupby(['user_id']).diff().rename({'video_day':'video_day_diff'},axis=1)],axis=1)
    video_day_diff_max = video_day_diff.groupby(['user_id'])['video_day_diff'].max().rename('video_day_diff_max').reset_index()
    video_day_diff_min = video_day_diff.groupby(['user_id'])['video_day_diff'].min().rename('video_day_diff_min').reset_index()
    video_day_diff_mean = video_day_diff.groupby(['user_id'])['video_day_diff'].mean().rename('video_day_diff_mean').reset_index()
    video_day_diff_std = video_day_diff.groupby(['user_id'])['video_day_diff'].std().rename('video_day_diff_std').reset_index()
    video_day_diff_kurt = video_day_diff.groupby(['user_id'])['video_day_diff'].agg(lambda x: pd.Series.kurt(x)).rename('video_day_diff_kurt').reset_index()
    video_day_diff_skew = video_day_diff.groupby(['user_id'])['video_day_diff'].skew().rename('video_day_diff_skew').reset_index()
    video_day_diff_last = video_day_diff.groupby(['user_id'])['video_day_diff'].last().rename('video_day_diff_last').reset_index()
    #最后一次video在哪一天
    video_last_day = video.groupby(['user_id'])['video_day'].max().rename('video_last_day').reset_index()
    
    #第一次video在哪一天
    video_first_day = video.groupby(['user_id'])['video_day'].min().rename('video_first_day').reset_index()
    
    #最后一次video到end day隔了多久
    video_rest_day = (video.groupby(['user_id'])['video_day'].max()-end_day).rename('video_rest_day').reset_index()
    video_rest_day['video_rest_day'] = -video_rest_day['video_rest_day']
    #按天统计特征
    video_count = video.groupby(['user_id','video_day']).agg({'video_day':'count'}).rename({'video_day':'video_count'},axis=1).reset_index()
    video_count_max = video_count.groupby(['user_id'])['video_count'].max().rename('video_count_max').reset_index()
    video_count_min = video_count.groupby(['user_id'])['video_count'].min().rename('video_count_min').reset_index()
    video_count_mean = video_count.groupby(['user_id'])['video_count'].mean().rename('video_count_mean').reset_index()
    video_count_std = video_count.groupby(['user_id'])['video_count'].std().rename('video_count_std').reset_index()
    video_count_kurt = video_count.groupby(['user_id'])['video_count'].agg(lambda x: pd.Series.kurt(x)).rename('video_count_kurt').reset_index()
    video_count_skew = video_count.groupby(['user_id'])['video_count'].skew().rename('video_count_skew').reset_index()
    video_count_last = video_count.groupby(['user_id'])['video_count'].last().rename('video_count_last').reset_index()
    video_day_mode = video_count.groupby(['user_id'])['video_count'].agg(lambda x: np.mean(pd.Series.mode(x))).rename('video_day_mode').reset_index()
    #最多video的是哪一天
    most_video_day = video_count.groupby('user_id').apply(lambda x: x[x.video_count==x.video_count.max()]).rename({'video_day':'most_video_day'},axis=1).drop('video_count',axis=1).groupby('user_id')['most_video_day'].max().reset_index()
    most_video_day_sub_end_day = (most_video_day-end_day).rename({'most_video_day':'most_video_day_sub_end_day'},axis=1)
    #去重后video相隔天数特征
    video_day_unique = video.groupby(['user_id','video_day']).agg({'user_id':'mean','video_day':'mean'}).rename({'video_day': 'video_day_unique'},axis=1)
    video_day_unique_diff = video_day_unique.groupby(['user_id']).diff().rename({'video_day_unique':'video_day_unique_diff'},axis=1).reset_index().drop('video_day',axis=1)
    video_day_unique_diff_max = video_day_unique_diff.groupby(['user_id'])['video_day_unique_diff'].max().rename('video_day_unique_diff_max').reset_index()
    video_day_unique_diff_min = video_day_unique_diff.groupby(['user_id'])['video_day_unique_diff'].min().rename('video_day_unique_diff_min').reset_index()
    video_day_unique_diff_mean = video_day_unique_diff.groupby(['user_id'])['video_day_unique_diff'].mean().rename('video_day_unique_diff_mean').reset_index()
    video_day_unique_diff_std = video_day_unique_diff.groupby(['user_id'])['video_day_unique_diff'].std().rename('video_day_unique_diff_std').reset_index()
    video_day_unique_diff_kurt = video_day_unique_diff.groupby(['user_id'])['video_day_unique_diff'].agg(lambda x: pd.Series.kurt(x)).rename('video_day_unique_diff_kurt').reset_index()
    video_day_unique_diff_skew = video_day_unique_diff.groupby(['user_id'])['video_day_unique_diff'].skew().rename('video_day_unique_diff_skew').reset_index()
    video_day_unique_diff_last = video_day_unique_diff.groupby(['user_id'])['video_day_unique_diff'].last().rename('video_day_unique_diff_last').reset_index()
    #MERGE
    feature = pd.merge(feature,video_total_count_all,how='left',on='user_id')
    feature = feature.rename({'video_total_count':'video_total_count_all'},axis=1)
    feature = pd.merge(feature,video_total_count_1,how='left',on='user_id')
    feature = feature.rename({'video_total_count':'video_total_count_1'},axis=1)
    feature = pd.merge(feature,video_total_count_2,how='left',on='user_id')
    feature = feature.rename({'video_total_count':'video_total_count_2'},axis=1)
    feature = pd.merge(feature,video_total_count_3,how='left',on='user_id')
    feature = feature.rename({'video_total_count':'video_total_count_3'},axis=1)
    feature = pd.merge(feature,video_total_count_5,how='left',on='user_id')
    feature = feature.rename({'video_total_count':'video_total_count_5'},axis=1)
    feature = pd.merge(feature,video_total_count_7,how='left',on='user_id')
    feature = feature.rename({'video_total_count':'video_total_count_7'},axis=1)
    feature = pd.merge(feature,video_total_count_10,how='left',on='user_id')
    feature = feature.rename({'video_total_count':'video_total_count_10'},axis=1)
    # 加入时间权重
    feature = pd.merge(feature,video_time_feat_all,how='left',on='user_id')
    feature = feature.rename({'video_time_feat':'video_time_feat_all'},axis=1)
    feature = pd.merge(feature,video_time_feat_1,how='left',on='user_id')
    feature = feature.rename({'video_time_feat':'video_time_feat_1'},axis=1)
    feature = pd.merge(feature,video_time_feat_2,how='left',on='user_id')
    feature = feature.rename({'video_time_feat':'video_time_feat_2'},axis=1)
    feature = pd.merge(feature,video_time_feat_3,how='left',on='user_id')
    feature = feature.rename({'video_time_feat':'video_time_feat_3'},axis=1)
    feature = pd.merge(feature,video_time_feat_5,how='left',on='user_id')
    feature = feature.rename({'video_time_feat':'video_time_feat_5'},axis=1)
    feature = pd.merge(feature,video_time_feat_7,how='left',on='user_id')
    feature = feature.rename({'video_time_feat':'video_time_feat_7'},axis=1)
    feature = pd.merge(feature,video_time_feat_10,how='left',on='user_id')
    feature = feature.rename({'video_time_feat':'video_time_feat_10'},axis=1)
    
    feature = pd.merge(feature,video_day_diff_max,how='left',on='user_id')
    feature = pd.merge(feature,video_day_diff_min,how='left',on='user_id')
    feature = pd.merge(feature,video_day_diff_mean,how='left',on='user_id')
    feature = pd.merge(feature,video_day_diff_std,how='left',on='user_id')
    feature = pd.merge(feature,video_day_diff_kurt,how='left',on='user_id')
    feature = pd.merge(feature,video_day_diff_skew,how='left',on='user_id')
    feature = pd.merge(feature,video_day_diff_last,how='left',on='user_id')
    feature = pd.merge(feature,video_day_unique_diff_max,how='left',on='user_id')
    feature = pd.merge(feature,video_day_unique_diff_min,how='left',on='user_id')
    feature = pd.merge(feature,video_day_unique_diff_mean,how='left',on='user_id')
    feature = pd.merge(feature,video_day_unique_diff_std,how='left',on='user_id')
    feature = pd.merge(feature,video_day_unique_diff_kurt,how='left',on='user_id')
    feature = pd.merge(feature,video_day_unique_diff_skew,how='left',on='user_id')
    feature = pd.merge(feature,video_day_unique_diff_last,how='left',on='user_id')
    feature = pd.merge(feature,video_last_day,how='left',on='user_id')
    feature = pd.merge(feature,video_first_day,how='left',on='user_id')
    feature = pd.merge(feature,video_rest_day,how='left',on='user_id')
    feature = pd.merge(feature,video_count_max,how='left',on='user_id')
    feature = pd.merge(feature,video_count_min,how='left',on='user_id')
    feature = pd.merge(feature,video_count_mean,how='left',on='user_id')
    feature = pd.merge(feature,video_count_std,how='left',on='user_id')
    feature = pd.merge(feature,video_count_kurt,how='left',on='user_id')
    feature = pd.merge(feature,video_count_skew,how='left',on='user_id')
    feature = pd.merge(feature,video_count_last,how='left',on='user_id')
    feature = pd.merge(feature,video_day_mode,how='left',on='user_id')
    feature = pd.merge(feature,most_video_day,how='left',on='user_id')
    feature = pd.merge(feature,most_video_day_sub_end_day,how='left',on='user_id')
    return feature

def activity_slide_window(activity):
    activity_total_count = activity[['user_id']].groupby(['user_id']).size().rename('activity_total_count').reset_index()
     # 加入权重信息
    activity['app_activity_log_flag'] = activity['app_activity_log_flag'].fillna(0)
    activity_date = np.max(activity['activity_day'])
    activity['weight'] = activity_date + 1- activity['activity_day']
    activity['weight'] = 1 / activity['weight']
    activity['app_activity_log_flag'] = activity['app_activity_log_flag'] * activity['weight']
    activity_time_feat = activity.groupby(['user_id'])['app_activity_log_flag'].sum().rename('windows_activity_feat').reset_index()
    
    #每个page个数
    page_count = activity.groupby(['user_id','page']).agg({'page':'count'}).rename({'page':'page_count'},axis=1).reset_index()
    page0_count = page_count[page_count.page==0].drop('page',axis=1).rename({'page_count':'page0_count'},axis=1)
    page1_count = page_count[page_count.page==1].drop('page',axis=1).rename({'page_count':'page1_count'},axis=1)
    page2_count = page_count[page_count.page==2].drop('page',axis=1).rename({'page_count':'page2_count'},axis=1)
    page3_count = page_count[page_count.page==3].drop('page',axis=1).rename({'page_count':'page3_count'},axis=1)
    page4_count = page_count[page_count.page==4].drop('page',axis=1).rename({'page_count':'page4_count'},axis=1)
    #每个page占比
    page_percent = pd.merge(activity_total_count,page0_count,how='left',on='user_id')
    page_percent = pd.merge(page_percent,page1_count,how='left',on='user_id')
    page_percent = pd.merge(page_percent,page2_count,how='left',on='user_id')
    page_percent = pd.merge(page_percent,page3_count,how='left',on='user_id')
    page_percent = pd.merge(page_percent,page4_count,how='left',on='user_id')
    
    page_percent['page0_pct'] = page_percent['page0_count']/page_percent['activity_total_count']
    page_percent['page1_pct'] = page_percent['page1_count']/page_percent['activity_total_count']
    page_percent['page2_pct'] = page_percent['page2_count']/page_percent['activity_total_count']
    page_percent['page3_pct'] = page_percent['page3_count']/page_percent['activity_total_count']
    page_percent['page4_pct'] = page_percent['page4_count']/page_percent['activity_total_count']
    page_percent = page_percent.drop('activity_total_count',axis=1)
    #每个action个数
    action_count = activity.groupby(['user_id','action']).agg({'action':'count'}).rename({'action':'action_count'},axis=1).reset_index()
    action0_count = action_count[action_count.action==0].drop('action',axis=1).rename({'action_count':'action0_count'},axis=1)
    action1_count = action_count[action_count.action==1].drop('action',axis=1).rename({'action_count':'action1_count'},axis=1)
    action2_count = action_count[action_count.action==2].drop('action',axis=1).rename({'action_count':'action2_count'},axis=1)
    action3_count = action_count[action_count.action==3].drop('action',axis=1).rename({'action_count':'action3_count'},axis=1)
#每个action占比
    action_percent = pd.merge(activity_total_count,action0_count,how='left',on='user_id')
    action_percent = pd.merge(action_percent,action1_count,how='left',on='user_id')
    action_percent = pd.merge(action_percent,action2_count,how='left',on='user_id')
    action_percent = pd.merge(action_percent,action3_count,how='left',on='user_id')
    action_percent['action0_pct'] = action_percent['action0_count']/action_percent['activity_total_count']
    action_percent['action1_pct'] = action_percent['action1_count']/action_percent['activity_total_count']
    action_percent['action2_pct'] = action_percent['action2_count']/action_percent['activity_total_count']
    action_percent['action3_pct'] = action_percent['action3_count']/action_percent['activity_total_count']
    action_percent = action_percent.drop('activity_total_count',axis=1)
    #看视频个数特征
    video_id_count = activity.groupby(['user_id','video_id']).agg({'user_id':'mean','video_id':'count'})
    video_id_max = video_id_count.groupby(['user_id'])['video_id'].max().rename('video_id_max').reset_index()
    video_id_min = video_id_count.groupby(['user_id'])['video_id'].min().rename('video_id_min').reset_index()
    video_id_mean = video_id_count.groupby(['user_id'])['video_id'].mean().rename('video_id_mean').reset_index()
    video_id_std = video_id_count.groupby(['user_id'])['video_id'].std().rename('video_id_std').reset_index()
    video_id_last = video_id_count.groupby(['user_id'])['video_id'].last().rename('video_id_last').reset_index()
    video_id = pd.merge(video_id_max,video_id_min,how='left',on='user_id')
    video_id = pd.merge(video_id,video_id_mean,how='left',on='user_id')
    video_id = pd.merge(video_id,video_id_std,how='left',on='user_id')
    video_id = pd.merge(video_id,video_id_last,how='left',on='user_id')
    #被看视频个数特征
    author_id_count = activity.groupby(['user_id','author_id']).agg({'user_id':'mean','author_id':'count'})
    author_id_max = author_id_count.groupby(['user_id'])['author_id'].max().rename('author_id_max').reset_index()
    author_id_min = author_id_count.groupby(['user_id'])['author_id'].min().rename('author_id_min').reset_index()
    author_id_mean = author_id_count.groupby(['user_id'])['author_id'].mean().rename('author_id_mean').reset_index()
    author_id_std = author_id_count.groupby(['user_id'])['author_id'].std().rename('author_id_std').reset_index()
    author_id_last = author_id_count.groupby(['user_id'])['author_id'].last().rename('author_id_last').reset_index()
    author_id = pd.merge(author_id_max,author_id_min,how='left',on='user_id')
    author_id = pd.merge(author_id,author_id_mean,how='left',on='user_id')
    author_id = pd.merge(author_id,author_id_std,how='left',on='user_id')
    author_id = pd.merge(author_id,author_id_last,how='left',on='user_id')
    #拍视频个数
    author_video_count = activity[['author_id','video_id']].drop_duplicates().groupby(['author_id']).agg({'video_id':'count'}).reset_index().rename({'author_id':'user_id','video_id':'author_video_count'},axis=1)
    #被每个action次数
    author_get_activity_count = activity.groupby('author_id').agg({'author_id': 'mean', 'activity_day': 'count'}).rename({'author_id':'user_id','activity_day':'author_get_activity_count'},axis=1)
    author_action_type_count = activity.groupby(['author_id','action']).agg({'action':'count'}).rename({'action':'author_action_type_count'},axis=1).reset_index()
    author_get_action0_count = author_action_type_count[author_action_type_count.action==0].drop('action',axis=1).rename({'author_id':'user_id','author_action_type_count':'author_get_action0_count'},axis=1)
    author_get_action1_count = author_action_type_count[author_action_type_count.action==1].drop('action',axis=1).rename({'author_id':'user_id','author_action_type_count':'author_get_action1_count'},axis=1)
    author_get_action2_count = author_action_type_count[author_action_type_count.action==2].drop('action',axis=1).rename({'author_id':'user_id','author_action_type_count':'author_get_action2_count'},axis=1)
    author_get_action3_count = author_action_type_count[author_action_type_count.action==3].drop('action',axis=1).rename({'author_id':'user_id','author_action_type_count':'author_get_action3_count'},axis=1)
#被每个action占比
    author_get_action_percent = pd.merge(author_get_activity_count,author_get_action0_count,how='left',on='user_id')
    author_get_action_percent = pd.merge(author_get_action_percent,author_get_action1_count,how='left',on='user_id')
    author_get_action_percent = pd.merge(author_get_action_percent,author_get_action2_count,how='left',on='user_id')
    author_get_action_percent = pd.merge(author_get_action_percent,author_get_action3_count,how='left',on='user_id')
    author_get_action_percent['author_get_action0_pct'] = author_get_action_percent['author_get_action0_count']/author_get_action_percent['author_get_activity_count']
    author_get_action_percent['author_get_action1_pct'] = author_get_action_percent['author_get_action1_count']/author_get_action_percent['author_get_activity_count']
    author_get_action_percent['author_get_action2_pct'] = author_get_action_percent['author_get_action2_count']/author_get_action_percent['author_get_activity_count']
    author_get_action_percent['author_get_action3_pct'] = author_get_action_percent['author_get_action3_count']/author_get_action_percent['author_get_activity_count']
    #被每个page次数
    author_page_type_count = activity.groupby(['author_id','page']).agg({'page':'count'}).rename({'page':'author_page_type_count'},axis=1).reset_index()
    author_get_page0_count = author_page_type_count[author_page_type_count.page==0].drop('page',axis=1).rename({'author_id':'user_id','author_page_type_count':'author_get_page0_count'},axis=1)
    author_get_page1_count = author_page_type_count[author_page_type_count.page==1].drop('page',axis=1).rename({'author_id':'user_id','author_page_type_count':'author_get_page1_count'},axis=1)
    author_get_page2_count = author_page_type_count[author_page_type_count.page==2].drop('page',axis=1).rename({'author_id':'user_id','author_page_type_count':'author_get_page2_count'},axis=1)
    author_get_page3_count = author_page_type_count[author_page_type_count.page==3].drop('page',axis=1).rename({'author_id':'user_id','author_page_type_count':'author_get_page3_count'},axis=1)
    author_get_page4_count = author_page_type_count[author_page_type_count.page==4].drop('page',axis=1).rename({'author_id':'user_id','author_page_type_count':'author_get_page4_count'},axis=1)
    #被每个page占比
    author_get_page_percent = pd.merge(author_get_activity_count,author_get_page0_count,how='left',on='user_id')
    author_get_page_percent = pd.merge(author_get_page_percent,author_get_page1_count,how='left',on='user_id')
    author_get_page_percent = pd.merge(author_get_page_percent,author_get_page2_count,how='left',on='user_id')
    author_get_page_percent = pd.merge(author_get_page_percent,author_get_page3_count,how='left',on='user_id')
    author_get_page_percent = pd.merge(author_get_page_percent,author_get_page4_count,how='left',on='user_id')
    author_get_page_percent['author_get_page0_pct'] = author_get_page_percent['author_get_page0_count']/author_get_page_percent['author_get_activity_count']
    author_get_page_percent['author_get_page1_pct'] = author_get_page_percent['author_get_page1_count']/author_get_page_percent['author_get_activity_count']
    author_get_page_percent['author_get_page2_pct'] = author_get_page_percent['author_get_page2_count']/author_get_page_percent['author_get_activity_count']
    author_get_page_percent['author_get_page3_pct'] = author_get_page_percent['author_get_page3_count']/author_get_page_percent['author_get_activity_count']
    author_get_page_percent['author_get_page4_pct'] = author_get_page_percent['author_get_page4_count']/author_get_page_percent['author_get_activity_count']
    return activity_total_count,page_percent,action_percent,video_id,author_id,author_video_count,author_get_activity_count,author_get_action_percent,author_get_page_percent,activity_time_feat

    
def get_activity_feature(activity):
    activity = activity.rename({'action_type':'action'},axis=1)
    feature = pd.DataFrame()
    feature['user_id'] = activity['user_id'].drop_duplicates()
    end_day = np.max(activity['activity_day'])
    #去重后activity相隔天数特征
    activity_day_unique = activity.groupby(['user_id','activity_day']).agg({'user_id':'mean','activity_day':'mean'})
    activity_day_diff = activity_day_unique.groupby(['user_id']).diff().rename({'activity_day':'activity_day_diff'},axis=1).reset_index().drop('activity_day',axis=1)
    activity_day_diff_max = activity_day_diff.groupby(['user_id'])['activity_day_diff'].max().rename('activity_day_diff_max').reset_index()
    activity_day_diff_min = activity_day_diff.groupby(['user_id'])['activity_day_diff'].min().rename('activity_day_diff_min').reset_index()
    activity_day_diff_mean = activity_day_diff.groupby(['user_id'])['activity_day_diff'].mean().rename('activity_day_diff_mean').reset_index()
    activity_day_diff_std = activity_day_diff.groupby(['user_id'])['activity_day_diff'].std().rename('activity_day_diff_std').reset_index()
    activity_day_diff_kurt = activity_day_diff.groupby(['user_id'])['activity_day_diff'].agg(lambda x: pd.Series.kurt(x)).rename('activity_day_diff_kurt').reset_index()
    activity_day_diff_skew = activity_day_diff.groupby(['user_id'])['activity_day_diff'].skew().rename('activity_day_diff_skew').reset_index()
    activity_day_diff_last = activity_day_diff.groupby(['user_id'])['activity_day_diff'].last().rename('activity_day_diff_last').reset_index()
    #最后一次activity在哪天
    activity_last_day = activity.groupby(['user_id'])['activity_day'].max().rename('activity_last_day').reset_index()
    
    #第一次activity在哪天
    activity_first_day = activity.groupby(['user_id'])['activity_day'].min().rename('activity_first_day').reset_index()

    
    #最后一次activity到end day隔了多久
    activity_rest_day = (activity_day_unique.groupby(['user_id'])['activity_day'].max()-end_day).rename('activity_rest_day').reset_index()
    activity_rest_day['activity_rest_day'] = -activity_rest_day['activity_rest_day']
    #按天统计特征
    activity_count = activity.groupby(['user_id','activity_day']).agg({'activity_day':'count'}).rename({'activity_day':'activity_count'},axis=1).reset_index()
    activity_count_max = activity_count.groupby(['user_id'])['activity_count'].max().rename('activity_count_max').reset_index()
    activity_count_min = activity_count.groupby(['user_id'])['activity_count'].min().rename('activity_count_min').reset_index()
    activity_count_mean = activity_count.groupby(['user_id'])['activity_count'].mean().rename('activity_count_mean').reset_index()
    activity_count_std = activity_count.groupby(['user_id'])['activity_count'].std().rename('activity_count_std').reset_index()
    activity_count_kurt = activity_count.groupby(['user_id'])['activity_count'].agg(lambda x: pd.Series.kurt(x)).rename('activity_count_kurt').reset_index()
    activity_count_skew = activity_count.groupby(['user_id'])['activity_count'].skew().rename('activity_count_skew').reset_index()
    activity_count_last = activity_count.groupby(['user_id'])['activity_count'].last().rename('activity_count_last').reset_index()
    #最多activity的是哪一天
    most_activity_day = activity_count.groupby('user_id').apply(lambda x: x[x.activity_count==x.activity_count.max()]).rename({'activity_day':'most_activity_day'},axis=1).drop('activity_count',axis=1).groupby('user_id')['most_activity_day'].max().reset_index()
    most_activity_day_sub_end_day = (most_activity_day-end_day).rename({'most_activity_day':'most_activity_day_sub_end_day'},axis=1)
    #activity相隔天数特征
    activity_count_diff = pd.concat([activity_count['user_id'],activity_count.groupby(['user_id']).diff().rename({'activity_count':'activity_count_diff'},axis=1)],axis=1).drop('activity_day',axis=1)
    activity_count_diff_max = activity_count_diff.groupby(['user_id'])['activity_count_diff'].max().rename('activity_count_diff_max').reset_index()
    activity_count_diff_min = activity_count_diff.groupby(['user_id'])['activity_count_diff'].min().rename('activity_count_diff_min').reset_index()
    activity_count_diff_mean = activity_count_diff.groupby(['user_id'])['activity_count_diff'].mean().rename('activity_count_diff_mean').reset_index()
    activity_count_diff_std = activity_count_diff.groupby(['user_id'])['activity_count_diff'].std().rename('activity_count_diff_std').reset_index()
    activity_count_diff_kurt = activity_count_diff.groupby(['user_id'])['activity_count_diff'].agg(lambda x: pd.Series.kurt(x)).rename('activity_count_diff_kurt').reset_index()
    activity_count_diff_skew = activity_count_diff.groupby(['user_id'])['activity_count_diff'].skew().rename('activity_count_diff_skew').reset_index()
    activity_count_diff_last = activity_count_diff.groupby(['user_id'])['activity_count_diff'].last().rename('activity_count_diff_last').reset_index()
    #每个page按天统计特征
    page_everyday_count = activity.groupby(['user_id','activity_day','page']).agg({'page':'count'}).rename({'page':'page_everyday_count'},axis=1).reset_index()
    page0_max = page_everyday_count[page_everyday_count.page==0].groupby(['user_id'])['page_everyday_count'].max().rename('page0_max').reset_index()
    page0_min = page_everyday_count[page_everyday_count.page==0].groupby(['user_id'])['page_everyday_count'].min().rename('page0_min').reset_index()
    page0_mean = page_everyday_count[page_everyday_count.page==0].groupby(['user_id'])['page_everyday_count'].mean().rename('page0_mean').reset_index()
    page0_std = page_everyday_count[page_everyday_count.page==0].groupby(['user_id'])['page_everyday_count'].std().rename('page0_std').reset_index()
    page0_kurt = page_everyday_count[page_everyday_count.page==0].groupby(['user_id'])['page_everyday_count'].agg(lambda x: pd.Series.kurt(x)).rename('page0_kurt').reset_index()
    page0_skew = page_everyday_count[page_everyday_count.page==0].groupby(['user_id'])['page_everyday_count'].skew().rename('page0_skew').reset_index()
    page0_last = page_everyday_count[page_everyday_count.page==0].groupby(['user_id'])['page_everyday_count'].last().rename('page0_last').reset_index()
    page1_max = page_everyday_count[page_everyday_count.page==1].groupby(['user_id'])['page_everyday_count'].max().rename('page1_max').reset_index()
    page1_min = page_everyday_count[page_everyday_count.page==1].groupby(['user_id'])['page_everyday_count'].min().rename('page1_min').reset_index()
    page1_mean = page_everyday_count[page_everyday_count.page==1].groupby(['user_id'])['page_everyday_count'].mean().rename('page1_mean').reset_index()
    page1_std = page_everyday_count[page_everyday_count.page==1].groupby(['user_id'])['page_everyday_count'].std().rename('page1_std').reset_index()
    page1_kurt = page_everyday_count[page_everyday_count.page==1].groupby(['user_id'])['page_everyday_count'].agg(lambda x: pd.Series.kurt(x)).rename('page1_kurt').reset_index()
    page1_skew = page_everyday_count[page_everyday_count.page==1].groupby(['user_id'])['page_everyday_count'].skew().rename('page1_skew').reset_index()
    page1_last = page_everyday_count[page_everyday_count.page==1].groupby(['user_id'])['page_everyday_count'].last().rename('page1_last').reset_index()
    page2_max = page_everyday_count[page_everyday_count.page==2].groupby(['user_id'])['page_everyday_count'].max().rename('page2_max').reset_index()
    page2_min = page_everyday_count[page_everyday_count.page==2].groupby(['user_id'])['page_everyday_count'].min().rename('page2_min').reset_index()
    page2_mean = page_everyday_count[page_everyday_count.page==2].groupby(['user_id'])['page_everyday_count'].mean().rename('page2_mean').reset_index()
    page2_std = page_everyday_count[page_everyday_count.page==2].groupby(['user_id'])['page_everyday_count'].std().rename('page2_std').reset_index()
    page2_kurt = page_everyday_count[page_everyday_count.page==2].groupby(['user_id'])['page_everyday_count'].agg(lambda x: pd.Series.kurt(x)).rename('page2_kurt').reset_index()
    page2_skew = page_everyday_count[page_everyday_count.page==2].groupby(['user_id'])['page_everyday_count'].skew().rename('page2_skew').reset_index()
    page2_last = page_everyday_count[page_everyday_count.page==2].groupby(['user_id'])['page_everyday_count'].last().rename('page2_last').reset_index()
    page3_max = page_everyday_count[page_everyday_count.page==3].groupby(['user_id'])['page_everyday_count'].max().rename('page3_max').reset_index()
    page3_min = page_everyday_count[page_everyday_count.page==3].groupby(['user_id'])['page_everyday_count'].min().rename('page3_min').reset_index()
    page3_mean = page_everyday_count[page_everyday_count.page==3].groupby(['user_id'])['page_everyday_count'].mean().rename('page3_mean').reset_index()
    page3_std = page_everyday_count[page_everyday_count.page==3].groupby(['user_id'])['page_everyday_count'].std().rename('page3_std').reset_index()
    page3_kurt = page_everyday_count[page_everyday_count.page==3].groupby(['user_id'])['page_everyday_count'].agg(lambda x: pd.Series.kurt(x)).rename('page3_kurt').reset_index()
    page3_skew = page_everyday_count[page_everyday_count.page==3].groupby(['user_id'])['page_everyday_count'].skew().rename('page3_skew').reset_index()
    page3_last = page_everyday_count[page_everyday_count.page==3].groupby(['user_id'])['page_everyday_count'].last().rename('page3_last').reset_index()
    page4_max = page_everyday_count[page_everyday_count.page==4].groupby(['user_id'])['page_everyday_count'].max().rename('page4_max').reset_index()
    page4_min = page_everyday_count[page_everyday_count.page==4].groupby(['user_id'])['page_everyday_count'].min().rename('page4_min').reset_index()
    page4_mean = page_everyday_count[page_everyday_count.page==4].groupby(['user_id'])['page_everyday_count'].mean().rename('page4_mean').reset_index()
    page4_std = page_everyday_count[page_everyday_count.page==4].groupby(['user_id'])['page_everyday_count'].std().rename('page4_std').reset_index()
    page4_kurt = page_everyday_count[page_everyday_count.page==4].groupby(['user_id'])['page_everyday_count'].agg(lambda x: pd.Series.kurt(x)).rename('page4_kurt').reset_index()
    page4_skew = page_everyday_count[page_everyday_count.page==4].groupby(['user_id'])['page_everyday_count'].skew().rename('page4_skew').reset_index()
    page4_last = page_everyday_count[page_everyday_count.page==4].groupby(['user_id'])['page_everyday_count'].last().rename('page4_last').reset_index()
    #每个action按天统计特征
    action_everyday_count = activity.groupby(['user_id','activity_day','action']).agg({'action':'count'}).rename({'action':'action_everyday_count'},axis=1).reset_index()
    action0_max = action_everyday_count[action_everyday_count.action==0].groupby(['user_id'])['action_everyday_count'].max().rename('action0_max').reset_index()
    action0_min = action_everyday_count[action_everyday_count.action==0].groupby(['user_id'])['action_everyday_count'].min().rename('action0_min').reset_index()
    action0_mean = action_everyday_count[action_everyday_count.action==0].groupby(['user_id'])['action_everyday_count'].mean().rename('action0_mean').reset_index()
    action0_std = action_everyday_count[action_everyday_count.action==0].groupby(['user_id'])['action_everyday_count'].std().rename('action0_std').reset_index()
    action0_kurt = action_everyday_count[action_everyday_count.action==0].groupby(['user_id'])['action_everyday_count'].agg(lambda x: pd.Series.kurt(x)).rename('action0_kurt').reset_index()
    action0_skew = action_everyday_count[action_everyday_count.action==0].groupby(['user_id'])['action_everyday_count'].skew().rename('action0_skew').reset_index()
    action0_last = action_everyday_count[action_everyday_count.action==0].groupby(['user_id'])['action_everyday_count'].last().rename('action0_last').reset_index()
    action1_max = action_everyday_count[action_everyday_count.action==1].groupby(['user_id'])['action_everyday_count'].max().rename('action1_max').reset_index()
    action1_min = action_everyday_count[action_everyday_count.action==1].groupby(['user_id'])['action_everyday_count'].min().rename('action1_min').reset_index()
    action1_mean = action_everyday_count[action_everyday_count.action==1].groupby(['user_id'])['action_everyday_count'].mean().rename('action1_mean').reset_index()
    action1_std = action_everyday_count[action_everyday_count.action==1].groupby(['user_id'])['action_everyday_count'].std().rename('action1_std').reset_index()
    action1_kurt = action_everyday_count[action_everyday_count.action==1].groupby(['user_id'])['action_everyday_count'].agg(lambda x: pd.Series.kurt(x)).rename('action1_kurt').reset_index()
    action1_skew = action_everyday_count[action_everyday_count.action==1].groupby(['user_id'])['action_everyday_count'].skew().rename('action1_skew').reset_index()
    action1_last = action_everyday_count[action_everyday_count.action==1].groupby(['user_id'])['action_everyday_count'].last().rename('action1_last').reset_index()
    action2_max = action_everyday_count[action_everyday_count.action==2].groupby(['user_id'])['action_everyday_count'].max().rename('action2_max').reset_index()
    action2_min = action_everyday_count[action_everyday_count.action==2].groupby(['user_id'])['action_everyday_count'].min().rename('action2_min').reset_index()
    action2_mean = action_everyday_count[action_everyday_count.action==2].groupby(['user_id'])['action_everyday_count'].mean().rename('action2_mean').reset_index()
    action2_std = action_everyday_count[action_everyday_count.action==2].groupby(['user_id'])['action_everyday_count'].std().rename('action2_std').reset_index()
    action2_kurt = action_everyday_count[action_everyday_count.action==2].groupby(['user_id'])['action_everyday_count'].agg(lambda x: pd.Series.kurt(x)).rename('action2_kurt').reset_index()
    action2_skew = action_everyday_count[action_everyday_count.action==2].groupby(['user_id'])['action_everyday_count'].skew().rename('action2_skew').reset_index()
    action2_last = action_everyday_count[action_everyday_count.action==2].groupby(['user_id'])['action_everyday_count'].last().rename('action2_last').reset_index()
    action3_max = action_everyday_count[action_everyday_count.action==3].groupby(['user_id'])['action_everyday_count'].max().rename('action3_max').reset_index()
    action3_min = action_everyday_count[action_everyday_count.action==3].groupby(['user_id'])['action_everyday_count'].min().rename('action3_min').reset_index()
    action3_mean = action_everyday_count[action_everyday_count.action==3].groupby(['user_id'])['action_everyday_count'].mean().rename('action3_mean').reset_index()
    action3_std = action_everyday_count[action_everyday_count.action==3].groupby(['user_id'])['action_everyday_count'].std().rename('action3_std').reset_index()
    action3_kurt = action_everyday_count[action_everyday_count.action==3].groupby(['user_id'])['action_everyday_count'].agg(lambda x: pd.Series.kurt(x)).rename('action3_kurt').reset_index()
    action3_skew = action_everyday_count[action_everyday_count.action==3].groupby(['user_id'])['action_everyday_count'].skew().rename('action3_skew').reset_index()
    action3_last = action_everyday_count[action_everyday_count.action==3].groupby(['user_id'])['action_everyday_count'].last().rename('action3_last').reset_index()
#滑窗
    activity_total_count_all,page_percent_all,action_percent_all,video_id_all,author_id_all,author_video_count_all,author_get_activity_count_all,author_get_action_percent_all,author_get_page_percent_all,activity_time_feat_all=activity_slide_window(activity)
    activity_total_count_1,page_percent_1,action_percent_1,video_id_1,author_id_1,author_video_count_1,author_get_activity_count_1,author_get_action_percent_1,author_get_page_percent_1,activity_time_feat_1=activity_slide_window(activity[(activity['activity_day']>=end_day) & (activity['activity_day']<=end_day)])
    activity_total_count_3,page_percent_3,action_percent_3,video_id_3,author_id_3,author_video_count_3,author_get_activity_count_3,author_get_action_percent_3,author_get_page_percent_3,activity_time_feat_3=activity_slide_window(activity[(activity['activity_day']>=end_day-2) & (activity['activity_day']<=end_day)])
    activity_total_count_7,page_percent_7,action_percent_7,video_id_7,author_id_7,author_video_count_7,author_get_activity_count_7,author_get_action_percent_7,author_get_page_percent_7,activity_time_feat_7=activity_slide_window(activity[(activity['activity_day']>=end_day-6) & (activity['activity_day']<=end_day)])
    #MERGE
    feature = pd.merge(feature,activity_day_diff_max,how='left',on='user_id')
    feature = pd.merge(feature,activity_day_diff_min,how='left',on='user_id')
    feature = pd.merge(feature,activity_day_diff_mean,how='left',on='user_id')
    feature = pd.merge(feature,activity_day_diff_std,how='left',on='user_id')
    feature = pd.merge(feature,activity_day_diff_kurt,how='left',on='user_id')
    feature = pd.merge(feature,activity_day_diff_skew,how='left',on='user_id')
    feature = pd.merge(feature,activity_day_diff_last,how='left',on='user_id')
    feature = pd.merge(feature,activity_last_day,how='left',on='user_id')
    feature = pd.merge(feature,activity_first_day,how='left',on='user_id')
    feature = pd.merge(feature,activity_rest_day,how='left',on='user_id')
    feature = pd.merge(feature,activity_total_count_all,how='left',on='user_id')
    feature = feature.rename({'activity_total_count':'activity_total_count_all'},axis=1)
    
    feature = pd.merge(feature,activity_total_count_1,how='left',on='user_id')
    feature = feature.rename({'activity_total_count':'activity_total_count_1'},axis=1)

    feature = pd.merge(feature,activity_total_count_3,how='left',on='user_id')
    feature = feature.rename({'activity_total_count':'activity_total_count_3'},axis=1)

    feature = pd.merge(feature,activity_total_count_7,how='left',on='user_id')
    feature = feature.rename({'activity_total_count':'activity_total_count_7'},axis=1)

    
    # 加入时间权重
    feature = pd.merge(feature,activity_time_feat_all,how='left',on='user_id')
    feature = feature.rename({'activity_time_feat':'activity_time_feat_all'},axis=1)
    
    feature = pd.merge(feature,activity_time_feat_1,how='left',on='user_id')
    feature = feature.rename({'activity_time_feat':'activity_time_feat_1'},axis=1)

    feature = pd.merge(feature,activity_time_feat_3,how='left',on='user_id')
    feature = feature.rename({'activity_time_feat':'activity_time_feat_3'},axis=1)

    feature = pd.merge(feature,activity_time_feat_7,how='left',on='user_id')
    feature = feature.rename({'activity_time_feat':'activity_time_feat_7'},axis=1)


    feature = pd.merge(feature,page_percent_all,how='left',on='user_id')
    feature = pd.merge(feature,action_percent_all,how='left',on='user_id')
    feature = pd.merge(feature,video_id_all,how='left',on='user_id')
    feature = pd.merge(feature,author_id_all,how='left',on='user_id')
    feature = pd.merge(feature,author_video_count_all,how='left',on='user_id')
    feature = feature.rename({'author_video_count':'author_video_count_all'},axis=1)
    feature = pd.merge(feature,author_get_activity_count_all,how='left',on='user_id')
    feature = pd.merge(feature,author_get_action_percent_all,how='left',on='user_id')
    feature = pd.merge(feature,author_get_page_percent_all,how='left',on='user_id')
    
    feature = pd.merge(feature,page_percent_1,how='left',on='user_id')
    feature = pd.merge(feature,action_percent_1,how='left',on='user_id')
    feature = pd.merge(feature,video_id_1,how='left',on='user_id')
    feature = pd.merge(feature,author_id_1,how='left',on='user_id')
    feature = pd.merge(feature,author_video_count_1,how='left',on='user_id')
    feature = feature.rename({'author_video_count':'author_video_count_1'},axis=1)
    feature = pd.merge(feature,author_get_activity_count_1,how='left',on='user_id')
    feature = pd.merge(feature,author_get_action_percent_1,how='left',on='user_id')
    feature = pd.merge(feature,author_get_page_percent_1,how='left',on='user_id')
    
    feature = pd.merge(feature,page_percent_3,how='left',on='user_id')
    feature = pd.merge(feature,action_percent_3,how='left',on='user_id')
    feature = pd.merge(feature,video_id_3,how='left',on='user_id')
    feature = pd.merge(feature,author_id_3,how='left',on='user_id')
    feature = pd.merge(feature,author_video_count_3,how='left',on='user_id')
    feature = feature.rename({'author_video_count':'author_video_count_3'},axis=1)
    feature = pd.merge(feature,author_get_activity_count_3,how='left',on='user_id')
    feature = pd.merge(feature,author_get_action_percent_3,how='left',on='user_id')
    feature = pd.merge(feature,author_get_page_percent_3,how='left',on='user_id')
    
    feature = pd.merge(feature,page_percent_7,how='left',on='user_id')
    feature = pd.merge(feature,action_percent_7,how='left',on='user_id')
    feature = pd.merge(feature,video_id_7,how='left',on='user_id')
    feature = pd.merge(feature,author_id_7,how='left',on='user_id')
    feature = pd.merge(feature,author_video_count_7,how='left',on='user_id')
    feature = feature.rename({'author_video_count':'author_video_count_7'},axis=1)
    feature = pd.merge(feature,author_get_activity_count_7,how='left',on='user_id')
    feature = pd.merge(feature,author_get_action_percent_7,how='left',on='user_id')
    feature = pd.merge(feature,author_get_page_percent_7,how='left',on='user_id')
    
    
    feature = pd.merge(feature,activity_count_max,how='left',on='user_id')
    feature = pd.merge(feature,activity_count_min,how='left',on='user_id')
    feature = pd.merge(feature,activity_count_mean,how='left',on='user_id')
    feature = pd.merge(feature,activity_count_std,how='left',on='user_id')
    feature = pd.merge(feature,activity_count_kurt,how='left',on='user_id')
    feature = pd.merge(feature,activity_count_skew,how='left',on='user_id')
    feature = pd.merge(feature,activity_count_last,how='left',on='user_id')
    feature = pd.merge(feature,activity_count_diff_max,how='left',on='user_id')
    feature = pd.merge(feature,activity_count_diff_min,how='left',on='user_id')
    feature = pd.merge(feature,activity_count_diff_mean,how='left',on='user_id')
    feature = pd.merge(feature,activity_count_diff_std,how='left',on='user_id')
    feature = pd.merge(feature,activity_count_diff_kurt,how='left',on='user_id')
    feature = pd.merge(feature,activity_count_diff_skew,how='left',on='user_id')
    feature = pd.merge(feature,activity_count_diff_last,how='left',on='user_id')
    feature = pd.merge(feature,page0_max,how='left',on='user_id')
    feature = pd.merge(feature,page0_min,how='left',on='user_id')
    feature = pd.merge(feature,page0_mean,how='left',on='user_id')
    feature = pd.merge(feature,page0_std,how='left',on='user_id')
    feature = pd.merge(feature,page0_kurt,how='left',on='user_id')
    feature = pd.merge(feature,page0_skew,how='left',on='user_id')
    feature = pd.merge(feature,page0_last,how='left',on='user_id')
    feature = pd.merge(feature,page1_max,how='left',on='user_id')
    feature = pd.merge(feature,page1_min,how='left',on='user_id')
    feature = pd.merge(feature,page1_mean,how='left',on='user_id')
    feature = pd.merge(feature,page1_std,how='left',on='user_id')
    feature = pd.merge(feature,page1_kurt,how='left',on='user_id')
    feature = pd.merge(feature,page1_skew,how='left',on='user_id')
    feature = pd.merge(feature,page1_last,how='left',on='user_id')
    feature = pd.merge(feature,page2_max,how='left',on='user_id')
    feature = pd.merge(feature,page2_min,how='left',on='user_id')
    feature = pd.merge(feature,page2_mean,how='left',on='user_id')
    feature = pd.merge(feature,page2_std,how='left',on='user_id')
    feature = pd.merge(feature,page2_kurt,how='left',on='user_id')
    feature = pd.merge(feature,page2_skew,how='left',on='user_id')
    feature = pd.merge(feature,page2_last,how='left',on='user_id')
    feature = pd.merge(feature,page3_max,how='left',on='user_id')
    feature = pd.merge(feature,page3_min,how='left',on='user_id')
    feature = pd.merge(feature,page3_mean,how='left',on='user_id')
    feature = pd.merge(feature,page3_std,how='left',on='user_id')
    feature = pd.merge(feature,page3_kurt,how='left',on='user_id')
    feature = pd.merge(feature,page3_skew,how='left',on='user_id')
    feature = pd.merge(feature,page3_last,how='left',on='user_id')
    feature = pd.merge(feature,page4_max,how='left',on='user_id')
    feature = pd.merge(feature,page4_min,how='left',on='user_id')
    feature = pd.merge(feature,page4_mean,how='left',on='user_id')
    feature = pd.merge(feature,page4_std,how='left',on='user_id')
    feature = pd.merge(feature,page4_kurt,how='left',on='user_id')
    feature = pd.merge(feature,page4_skew,how='left',on='user_id')
    feature = pd.merge(feature,page4_last,how='left',on='user_id')
    feature = pd.merge(feature,action0_max,how='left',on='user_id')
    feature = pd.merge(feature,action0_min,how='left',on='user_id')
    feature = pd.merge(feature,action0_mean,how='left',on='user_id')
    feature = pd.merge(feature,action0_std,how='left',on='user_id')
    feature = pd.merge(feature,action0_kurt,how='left',on='user_id')
    feature = pd.merge(feature,action0_skew,how='left',on='user_id')
    feature = pd.merge(feature,action0_last,how='left',on='user_id')
    feature = pd.merge(feature,action1_max,how='left',on='user_id')
    feature = pd.merge(feature,action1_min,how='left',on='user_id')
    feature = pd.merge(feature,action1_mean,how='left',on='user_id')
    feature = pd.merge(feature,action1_std,how='left',on='user_id')
    feature = pd.merge(feature,action1_kurt,how='left',on='user_id')
    feature = pd.merge(feature,action1_skew,how='left',on='user_id')
    feature = pd.merge(feature,action1_last,how='left',on='user_id')
    feature = pd.merge(feature,action2_max,how='left',on='user_id')
    feature = pd.merge(feature,action2_min,how='left',on='user_id')
    feature = pd.merge(feature,action2_mean,how='left',on='user_id')
    feature = pd.merge(feature,action2_std,how='left',on='user_id')
    feature = pd.merge(feature,action2_kurt,how='left',on='user_id')
    feature = pd.merge(feature,action2_skew,how='left',on='user_id')
    feature = pd.merge(feature,action2_last,how='left',on='user_id')
    feature = pd.merge(feature,action3_max,how='left',on='user_id')
    feature = pd.merge(feature,action3_min,how='left',on='user_id')
    feature = pd.merge(feature,action3_mean,how='left',on='user_id')
    feature = pd.merge(feature,action3_std,how='left',on='user_id')
    feature = pd.merge(feature,action3_kurt,how='left',on='user_id')
    feature = pd.merge(feature,action3_skew,how='left',on='user_id')
    feature = pd.merge(feature,action3_last,how='left',on='user_id')
    feature = pd.merge(feature,most_activity_day,how='left',on='user_id')
    feature = pd.merge(feature,most_activity_day_sub_end_day,how='left',on='user_id')
    return feature
    
def deal_feature(path, user_id):
    register = pd.read_csv(path + '/register.csv')
    launch = pd.read_csv(path + '/launch.csv')
    video = pd.read_csv(path + '/video.csv')
    activity = pd.read_csv(path + '/activity.csv')
    feature = pd.DataFrame()
    feature['user_id'] = user_id
    
    print('getting activity feature...')
    activity['app_activity_log_flag'] = 1
    activity_feature = get_activity_feature(activity)
    feature = pd.merge(feature, activity_feature, on='user_id', how='left')
    del activity_feature
    gc.collect()
    print('activity表提取特征完毕!')
    
    print('getting register feature...')
    register_feature = get_register_feature(register)
    feature = pd.merge(feature, register_feature, on='user_id', how='left')
    del register_feature
    gc.collect()
    print('register表提取特征完毕!')

    
    print('getting launch feature...')
    launch['app_launch_log_flag'] = 1
    launch_feature = get_launch_feature(launch)
    feature = pd.merge(feature, launch_feature, on='user_id', how='left')
    del launch_feature
    gc.collect()
    print('launch表提取特征完毕!')
    
    
    print('getting video feature...')
    video['app_video_log_flag'] = 1
    video_feature = get_video_feature(video)
    feature = pd.merge(feature, video_feature, on='user_id', how='left')
    del video_feature
    gc.collect()
    print('create表提取特征完毕!')
  

    feature['bayes_f1'] = feature['windows_video_feat'] / (feature['windows_launch_feat'] + 0.00001)
    # feature['bayes_f2'] = feature['windows_activity_feat'] / (feature['windows_launch_feat'] + 0.00001)

    # feature['bayes_f3'] = feature['windows_launch_feat_x.2'] / (feature['windows_launch_feat'] + 0.00001)
    # feature['bayes_f4'] = (feature['windows_launch_feat'] - feature['windows_launch_feat_x.2']) / (feature['windows_launch_feat'] + 0.00001)

    #最后一次与注册日之差
    feature['last_launch_sub_register'] = feature['launch_last_day'] - feature['register_day']
    feature['last_video_sub_register'] = feature['video_last_day'] - feature['register_day']
    feature['last_activity_sub_register'] = feature['activity_last_day'] - feature['register_day']
    
    #第一次与注册日之差
    feature['first_launch_sub_register'] = feature['launch_first_day'] - feature['register_day']
    feature['first_video_sub_register'] = feature['video_first_day'] - feature['register_day']
    feature['first_activity_sub_register'] = feature['activity_first_day'] - feature['register_day']
    
    #滑窗 被看video数占拍视频数
    feature['author_video_pct_all'] = feature['author_video_count_all']/feature['video_total_count_all']
    feature['author_video_pct_1'] = feature['author_video_count_1']/feature['video_total_count_1']
    # feature['author_video_pct_2'] = feature['author_video_count_2']/feature['video_total_count_2']
    feature['author_video_pct_3'] = feature['author_video_count_3']/feature['video_total_count_3']
    # feature['author_video_pct_5'] = feature['author_video_count_5']/feature['video_total_count_5']
    feature['author_video_pct_7'] = feature['author_video_count_7']/feature['video_total_count_7']
    # feature['author_video_pct_10'] = feature['author_video_count_10']/feature['video_total_count_10']
    #平均 除以最大可能天数
    feature['mean_launch_total_count_all'] = feature['launch_total_count_all']/feature['divisor_day']
    feature['mean_video_total_count_all'] = feature['video_total_count_all']/feature['divisor_day']
    feature['mean_activity_total_count_all'] = feature['activity_total_count_all']/feature['divisor_day']
    # feature['mean_page0_count'] = feature['page0_count']/feature['divisor_day']
    # feature['mean_page1_count'] = feature['page1_count']/feature['divisor_day']
    # feature['mean_page2_count'] = feature['page2_count']/feature['divisor_day']
    # feature['mean_page3_count'] = feature['page3_count']/feature['divisor_day']
    # feature['mean_page4_count'] = feature['page4_count']/feature['divisor_day']
    # feature['mean_action0_count'] = feature['action0_count']/feature['divisor_day']
    # feature['mean_action1_count'] = feature['action1_count']/feature['divisor_day']
    # feature['mean_action2_count'] = feature['action2_count']/feature['divisor_day']
    # feature['mean_action3_count'] = feature['action3_count']/feature['divisor_day']

    #规则 7天之内启动过一次的为活跃用户
    feature.loc[(feature['launch_total_count_all']==1) & (feature['register_rest_day']>7),'launch_only_once']=1

    feature = feature.fillna(0)
    return feature


def get_data_feature():
    print('Feature engineering...')
    print('Getting train data 1 ...')
    train_label_1 = get_train_label(dataset_1_feat_dir,dataset_1_label_dir)
    data_1 = deal_feature(dataset_1_feat_dir,train_label_1['user_id'])
    data_1['label'] = train_label_1['label']
    data_1.to_csv(train_path1,index=False)
    
    print('Getting train data 2 ...')
    train_label_2 = get_train_label(dataset_2_feat_dir,dataset_2_label_dir)
    data_2 = deal_feature(dataset_2_feat_dir,train_label_2['user_id'])
    data_2['label'] = train_label_2['label']
    data_2.to_csv(train_path2,index=False)
    
    print('Getting train data 3 ...')
    train_label_3 = get_train_label(dataset_3_feat_dir,dataset_3_label_dir)
    data_3 = deal_feature(dataset_3_feat_dir,train_label_3['user_id'])
    data_3['label'] = train_label_3['label']
    data_3.to_csv(train_path3,index=False)

    
    train_data = pd.concat([data_1, data_2, data_3],axis = 0)
    train_data.to_csv(train_path,index=False)
    # 释放内存
    del data_1,data_2,data_3,train_data
    gc.collect()
    
    print('Getting test data...')
    test_id = get_test_id(dataset_4_feat_dir)
    test_data = deal_feature(dataset_4_feat_dir,test_id['user_id'])
    test_data.to_csv(test_path,index=False)
    
    # 释放内存
    del test_data
    gc.collect()

time_start = datetime.now()
print('Start time:',time_start.strftime('%Y-%m-%d %H:%M:%S'))
get_data_feature()
time_end = datetime.now()
print('End time:',time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('Total time:',"%.2f" % ((time_end-time_start).seconds/60),'minutes')