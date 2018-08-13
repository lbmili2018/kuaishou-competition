# -*- coding: utf-8 -*-

'''步骤1：滑动窗口划分数据'''
import pandas as pd
import numpy as np
from datetime import datetime

print('Generating datasets...')
input_dir = '/mnt/datasets/fusai/'

register = pd.read_csv(input_dir + 'user_register_log.txt', sep='\t',
                    names=['user_id','register_day','register_type','device_type'],
                    dtype={0: np.uint32, 1: np.uint8, 2: np.uint8, 3: np.uint16})
launch = pd.read_csv(input_dir + 'app_launch_log.txt', sep='\t',
                  names=['user_id', 'launch_day'],
                  dtype={0: np.uint32, 1: np.uint8})
video = pd.read_csv(input_dir + 'video_create_log.txt', sep='\t',
                  names=['user_id', 'video_day'],
                  dtype={0: np.uint32, 1: np.uint8})
activity = pd.read_csv(input_dir + 'user_activity_log.txt', sep='\t',
                    names=['user_id','activity_day','page','video_id','author_id','action_type'],
                    dtype={0: np.uint32, 1: np.uint8, 2: np.uint8, 3: np.uint32, 4: np.uint32, 5: np.uint8})

dataset_1_feat_dir = '/home/kesci/dataset_1_feat'
dataset_1_label_dir = '/home/kesci/dataset_1_label'
dataset_2_feat_dir = '/home/kesci/dataset_2_feat'
dataset_2_label_dir = '/home/kesci/dataset_2_label'
dataset_3_feat_dir = '/home/kesci/dataset_3_feat'
dataset_3_label_dir = '/home/kesci/dataset_3_label'
dataset_4_feat_dir = '/home/kesci/dataset_4_feat'


def cut_data_on_time(output_path,begin_day,end_day):
    temp_register = register[(register['register_day'] >= 1) & (register['register_day'] <= end_day)]
    temp_launch = launch[(launch['launch_day'] >= begin_day) & (launch['launch_day'] <= end_day)]
    temp_video = video[(video['video_day'] >= begin_day) & (video['video_day'] <= end_day)]
    temp_activity = activity[(activity['activity_day'] >= begin_day) & (activity['activity_day'] <= end_day)]
    
    temp_register.to_csv(output_path + '/register.csv',index=False)
    del temp_register
    temp_launch.to_csv(output_path + '/launch.csv',index=False)
    del temp_launch
    temp_video.to_csv(output_path + '/video.csv',index=False)
    del temp_video
    temp_activity.to_csv(output_path + '/activity.csv',index=False)
    del temp_activity
    
def generate_dataset():
    print('Cutting train data set 1 ...')
    begin_day = 1
    end_day = 9
    cut_data_on_time(dataset_1_feat_dir,begin_day,end_day)
    begin_day = 10
    end_day = 16
    cut_data_on_time(dataset_1_label_dir,begin_day,end_day)
    
    print('Cutting train data set 2 ...')
    begin_day = 1
    end_day = 16
    cut_data_on_time(dataset_2_feat_dir,begin_day,end_day)
    begin_day = 17
    end_day = 23
    cut_data_on_time(dataset_2_label_dir,begin_day,end_day)
    
    print('Cutting train data set 3 ...')
    begin_day = 1
    end_day = 23
    cut_data_on_time(dataset_3_feat_dir,begin_day,end_day)
    begin_day = 24
    end_day = 30
    cut_data_on_time(dataset_3_label_dir,begin_day,end_day)

    print('Cutting test data set...')
    begin_day = 1
    end_day = 30
    cut_data_on_time(dataset_4_feat_dir,begin_day,end_day)

generate_dataset()
print('Dataset generated.')
