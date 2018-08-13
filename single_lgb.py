import os
import sys
import time
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import datetime
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")


def feature_selection(feature_mode,importance_threshold,R_threshold,P_threshold,train,test,test_userid):
    # train_path = '/home/kesci/train_480.csv'
    # test_path = '/home/kesci/test_480.csv'
    
    # train = pd.read_csv(train_path)
    # test = pd.read_csv(test_path)
    # test_userid = test.pop('user_id')
    if feature_mode == 1:
        print('Loading all the features and label...')
        train_feature = train.drop(['user_id','label'],axis=1)
        train_label = train['label']
        online_test_feature = test
        print('特征数：'+ str(train_feature.columns.size))
    elif feature_mode == 2:
        print('Loading result-based important features and label...')
        feature_list = train.columns.values.tolist()
        feature_list.remove('user_id')
        feature_list.remove('label')
        feature_list = np.array(feature_list)
        feature_importance = [154, 11, 318, 148, 80, 57, 100, 222, 134, 104, 167, 125, 71, 61, 328, 20, 118, 194, 91, 66, 79, 75, 37, 115, 210, 129, 165, 64, 131, 51, 38, 23, 242, 167, 149, 144, 86, 14, 247, 128, 18, 88, 20, 117, 106, 42, 3, 11, 0, 14, 8, 0, 2, 18, 21, 0, 13, 0, 2, 18, 9, 25, 26, 2, 16, 9, 52, 29, 102, 166, 79, 19, 10, 40, 63, 52, 41, 14, 301, 26, 28, 4, 71, 35, 46, 17, 41, 5, 65, 66, 5, 25, 53, 91, 64, 39, 1, 5, 0, 13, 2, 0, 0, 7, 15, 0, 0, 0, 0, 3, 1, 8, 4, 0, 10, 0, 3, 14, 88, 126, 63, 36, 22, 73, 84, 43, 66, 34, 83, 16, 21, 10, 105, 56, 59, 31, 28, 0, 63, 65, 7, 46, 16, 61, 52, 28, 6, 8, 0, 7, 2, 0, 0, 12, 4, 0, 20, 0, 0, 3, 1, 22, 9, 0, 13, 3, 9, 13, 85, 115, 77, 47, 27, 114, 87, 67, 107, 66, 146, 42, 24, 8, 126, 145, 80, 29, 50, 4, 121, 89, 14, 50, 18, 59, 75, 27, 4, 9, 1, 9, 5, 0, 1, 15, 19, 0, 2, 0, 0, 5, 1, 6, 30, 6, 62, 10, 21, 53, 41, 55, 73, 68, 35, 82, 66, 89, 92, 126, 64, 78, 100, 125, 67, 114, 73, 66, 62, 72, 92, 46, 41, 70, 70, 64, 66, 60, 39, 56, 52, 50, 39, 56, 46, 49, 74, 47, 51, 74, 115, 63, 19, 21, 35, 29, 26, 8, 12, 60, 77, 56, 51, 50, 59, 55, 32, 56, 55, 55, 50, 57, 34, 27, 21, 48, 36, 27, 34, 16, 11, 9, 17, 14, 12, 14, 5, 185, 23, 166, 9, 613, 947, 248, 625, 17, 225, 147, 30, 12, 14, 53, 16, 0, 29, 0, 0, 105, 9, 62, 2, 32, 59, 74, 60, 135, 1096, 0, 0, 2, 97, 170, 394, 66, 66, 227, 182, 247, 120, 219, 258, 2, 205, 0, 0, 0, 0, 0, 0, 0, 0, 25, 2, 23, 28, 17, 5, 29, 13, 17, 31, 4, 21, 13, 15, 17, 16, 13, 7, 32, 16, 29, 32, 7, 9, 4, 9, 2, 3, 6, 2, 17, 48, 37, 4, 5, 7, 13, 25, 4, 7, 10, 13, 3, 106, 142, 10, 17, 0, 17, 63, 7, 7, 2, 4, 271, 37, 210, 5]
        feature_importance = np.array(feature_importance)
        feature_importance_check = np.vstack((feature_list,feature_importance)).T
        feature_importance_check = pd.DataFrame(feature_importance_check)
        feature_importance_check[1] = feature_importance_check[1].astype(int)
        used_feature = feature_list[feature_importance>=importance_threshold]
        train_feature = train[used_feature]
        train_label = train['label']
        online_test_feature = test[used_feature]
        print('importance_threshold:'+str(importance_threshold)+' 特征数：'+ str(train_feature.columns.size))
    elif feature_mode == 3:
        print('Loading Pearson important features and label...')
        train_fill = train.fillna(0)
        feature_list = train.columns.values.tolist()
        feature_list.remove('user_id')
        feature_list.remove('label')
        pearson = []
        for name in feature_list:
            pearson.append(pearsonr(train['label'], train_fill[name]))
        pearson = pd.DataFrame(pearson).rename({0:'R_value',1:'P_value'},axis=1)
        pearson['Feature_name'] = feature_list
        # print('所有特征：'+ str(list(train.columns.values)))
        # print('R_value：'+ str(list(pearson['R_value'])))
        # print('P_value：'+ str(list(pearson['P_value'])))
        used_feature = pearson[(pearson.P_value<=P_threshold) & ((pearson.R_value>=R_threshold)|(pearson.R_value<=-R_threshold))]
        used_feature = used_feature.Feature_name.tolist()
        train_feature = train[used_feature]
        train_label = train['label']
        online_test_feature = test[used_feature]
        print('R_threshold:'+str(R_threshold)+' P_threshold:'+str(P_threshold)+' 特征数：'+ str(train_feature.columns.size))
    elif feature_mode == 4:
        print('Loading selected features and label...')
        feature_list = ['register_day', 'register_day_sub_begin_day', 'register_type', 'device_type', 'register_rest_day', 'divisor_day', 'launch_slide_count_all', 'launch_slide_count_1', 'launch_slide_count_2', 'launch_slide_count_3', 'launch_slide_count_5', 'launch_slide_count_7', 'launch_slide_count_10', 'launch_day_kurt', 'launch_day_max', 'launch_day_skew', 'launch_day_mean', 'launch_day_std', 'launch_day_min', 'launch_rest_day', 'launch_day_diff_max', 'launch_day_diff_skew', 'launch_day_diff_min', 'launch_day_diff_kurt', 'launch_day_diff_std', 'launch_day_diff_mean', 'launch_day_diff_last', 'video_slide_count_all', 'video_slide_count_1', 'video_slide_count_2', 'video_slide_count_3', 'video_slide_count_5', 'video_slide_count_7', 'video_slide_count_10', 'video_day_std', 'video_day_skew', 'video_day_min', 'video_day_max', 'video_day_kurt', 'video_day_mean', 'video_rest_day', 'video_day_mode', 'most_video_day', 'most_video_day_sub_end_day', 'video_day_diff_mean', 'video_day_diff_kurt', 'video_day_diff_max', 'video_day_diff_min', 'video_day_diff_skew', 'video_day_diff_std', 'video_day_diff_last', 'video_day_unique_std', 'video_day_unique_mean', 'video_day_unique_kurt', 'video_day_unique_skew', 'video_day_unique_diff_max', 'video_day_unique_diff_mean', 'video_day_unique_diff_kurt', 'video_day_unique_diff_min', 'video_day_unique_diff_std', 'video_day_unique_diff_skew', 'video_day_unique_diff_last', 'video_everyday_count_max', 'video_everyday_count_min', 'video_everyday_count_std', 'video_everyday_count_mean', 'video_everyday_count_skew', 'video_everyday_count_kurt', 'video_everyday_count_last', 'activity_slide_count_all', 'page0_slide_count_all', 'page1_slide_count_all', 'page2_slide_count_all', 'page3_slide_count_all', 'page4_slide_count_all', 'page0_slide_pct_all', 'page1_slide_pct_all', 'page2_slide_pct_all', 'page3_slide_pct_all', 'page4_slide_pct_all', 'activity_slide_count_1', 'page0_slide_count_1', 'page1_slide_count_1', 'page2_slide_count_1', 'page3_slide_count_1', 'page4_slide_count_1', 'page0_slide_pct_1', 'page1_slide_pct_1', 'page2_slide_pct_1', 'page3_slide_pct_1', 'page4_slide_pct_1', 'activity_slide_count_2', 'page0_slide_count_2', 'page1_slide_count_2', 'page2_slide_count_2', 'page3_slide_count_2', 'page4_slide_count_2', 'page0_slide_pct_2', 'page1_slide_pct_2', 'page2_slide_pct_2', 'page3_slide_pct_2', 'page4_slide_pct_2', 'activity_slide_count_3', 'page0_slide_count_3', 'page1_slide_count_3', 'page2_slide_count_3', 'page3_slide_count_3', 'page4_slide_count_3', 'page0_slide_pct_3', 'page1_slide_pct_3', 'page2_slide_pct_3', 'page3_slide_pct_3', 'page4_slide_pct_3', 'activity_slide_count_5', 'page0_slide_count_5', 'page1_slide_count_5', 'page2_slide_count_5', 'page3_slide_count_5', 'page4_slide_count_5', 'page0_slide_pct_5', 'page1_slide_pct_5', 'page2_slide_pct_5', 'page3_slide_pct_5', 'page4_slide_pct_5', 'activity_slide_count_7', 'page0_slide_count_7', 'page1_slide_count_7', 'page2_slide_count_7', 'page3_slide_count_7', 'page4_slide_count_7', 'page0_slide_pct_7', 'page1_slide_pct_7', 'page2_slide_pct_7', 'page3_slide_pct_7', 'page4_slide_pct_7', 'activity_slide_count_10', 'page0_slide_count_10', 'page1_slide_count_10', 'page2_slide_count_10', 'page3_slide_count_10', 'page4_slide_count_10', 'page0_slide_pct_10', 'page1_slide_pct_10', 'page2_slide_pct_10', 'page3_slide_pct_10', 'page4_slide_pct_10', 'action0_slide_count_all', 'action1_slide_count_all', 'action2_slide_count_all', 'action3_slide_count_all', 'action4_slide_count_all', 'action5_slide_count_all', 'action0_slide_pct_all', 'action1_slide_pct_all', 'action2_slide_pct_all', 'action3_slide_pct_all', 'action4_slide_pct_all', 'action5_slide_pct_all', 'action0_slide_count_1', 'action1_slide_count_1', 'action2_slide_count_1', 'action3_slide_count_1', 'action4_slide_count_1', 'action5_slide_count_1', 'action0_slide_pct_1', 'action1_slide_pct_1', 'action2_slide_pct_1', 'action3_slide_pct_1', 'action4_slide_pct_1', 'action5_slide_pct_1', 'action0_slide_count_2', 'action1_slide_count_2', 'action2_slide_count_2', 'action3_slide_count_2', 'action4_slide_count_2', 'action5_slide_count_2', 'action0_slide_pct_2', 'action1_slide_pct_2', 'action2_slide_pct_2', 'action3_slide_pct_2', 'action4_slide_pct_2', 'action5_slide_pct_2', 'action0_slide_count_3', 'action1_slide_count_3', 'action2_slide_count_3', 'action3_slide_count_3', 'action4_slide_count_3', 'action5_slide_count_3', 'action0_slide_pct_3', 'action1_slide_pct_3', 'action2_slide_pct_3', 'action3_slide_pct_3', 'action4_slide_pct_3', 'action5_slide_pct_3', 'action0_slide_count_5', 'action1_slide_count_5', 'action2_slide_count_5', 'action3_slide_count_5', 'action4_slide_count_5', 'action5_slide_count_5', 'action0_slide_pct_5', 'action1_slide_pct_5', 'action2_slide_pct_5', 'action3_slide_pct_5', 'action4_slide_pct_5', 'action5_slide_pct_5', 'action0_slide_count_7', 'action1_slide_count_7', 'action2_slide_count_7', 'action3_slide_count_7', 'action4_slide_count_7', 'action5_slide_count_7', 'action0_slide_pct_7', 'action1_slide_pct_7', 'action2_slide_pct_7', 'action3_slide_pct_7', 'action4_slide_pct_7', 'action5_slide_pct_7', 'action0_slide_count_10', 'action1_slide_count_10', 'action2_slide_count_10', 'action3_slide_count_10', 'action4_slide_count_10', 'action5_slide_count_10', 'action0_slide_pct_10', 'action1_slide_pct_10', 'action2_slide_pct_10', 'action3_slide_pct_10', 'action4_slide_pct_10', 'action5_slide_pct_10', 'get_activity_slide_count_all', 'get_page0_slide_count_all', 'get_page1_slide_count_all', 'get_page2_slide_count_all', 'get_page3_slide_count_all', 'get_page4_slide_count_all', 'get_page0_slide_pct_all', 'get_page1_slide_pct_all', 'get_page2_slide_pct_all', 'get_page3_slide_pct_all', 'get_page4_slide_pct_all', 'get_activity_slide_count_1', 'get_page0_slide_count_1', 'get_page1_slide_count_1', 'get_page2_slide_count_1', 'get_page3_slide_count_1', 'get_page4_slide_count_1', 'get_page0_slide_pct_1', 'get_page1_slide_pct_1', 'get_page2_slide_pct_1', 'get_page3_slide_pct_1', 'get_page4_slide_pct_1', 'get_activity_slide_count_2', 'get_page0_slide_count_2', 'get_page1_slide_count_2', 'get_page2_slide_count_2', 'get_page3_slide_count_2', 'get_page4_slide_count_2', 'get_page0_slide_pct_2', 'get_page1_slide_pct_2', 'get_page2_slide_pct_2', 'get_page3_slide_pct_2', 'get_page4_slide_pct_2', 'get_activity_slide_count_3', 'get_page0_slide_count_3', 'get_page1_slide_count_3', 'get_page2_slide_count_3', 'get_page3_slide_count_3', 'get_page4_slide_count_3', 'get_page0_slide_pct_3', 'get_page1_slide_pct_3', 'get_page2_slide_pct_3', 'get_page3_slide_pct_3', 'get_page4_slide_pct_3', 'get_activity_slide_count_5', 'get_page0_slide_count_5', 'get_page1_slide_count_5', 'get_page2_slide_count_5', 'get_page3_slide_count_5', 'get_page4_slide_count_5', 'get_page0_slide_pct_5', 'get_page1_slide_pct_5', 'get_page2_slide_pct_5', 'get_page3_slide_pct_5', 'get_page4_slide_pct_5', 'get_activity_slide_count_7', 'get_page0_slide_count_7', 'get_page1_slide_count_7', 'get_page2_slide_count_7', 'get_page3_slide_count_7', 'get_page4_slide_count_7', 'get_page0_slide_pct_7', 'get_page1_slide_pct_7', 'get_page2_slide_pct_7', 'get_page3_slide_pct_7', 'get_page4_slide_pct_7', 'get_activity_slide_count_10', 'get_page0_slide_count_10', 'get_page1_slide_count_10', 'get_page2_slide_count_10', 'get_page3_slide_count_10', 'get_page4_slide_count_10', 'get_page0_slide_pct_10', 'get_page1_slide_pct_10', 'get_page2_slide_pct_10', 'get_page3_slide_pct_10', 'get_page4_slide_pct_10', 'get_action0_slide_count_all', 'get_action1_slide_count_all', 'get_action2_slide_count_all', 'get_action3_slide_count_all', 'get_action4_slide_count_all', 'get_action5_slide_count_all', 'get_action0_slide_pct_all', 'get_action1_slide_pct_all', 'get_action2_slide_pct_all', 'get_action3_slide_pct_all', 'get_action4_slide_pct_all', 'get_action5_slide_pct_all', 'get_action0_slide_count_1', 'get_action1_slide_count_1', 'get_action2_slide_count_1', 'get_action3_slide_count_1', 'get_action4_slide_count_1', 'get_action5_slide_count_1', 'get_action0_slide_pct_1', 'get_action1_slide_pct_1', 'get_action2_slide_pct_1', 'get_action3_slide_pct_1', 'get_action4_slide_pct_1', 'get_action5_slide_pct_1', 'get_action0_slide_count_2', 'get_action1_slide_count_2', 'get_action2_slide_count_2', 'get_action3_slide_count_2', 'get_action4_slide_count_2', 'get_action5_slide_count_2', 'get_action0_slide_pct_2', 'get_action1_slide_pct_2', 'get_action2_slide_pct_2', 'get_action3_slide_pct_2', 'get_action4_slide_pct_2', 'get_action5_slide_pct_2', 'get_action0_slide_count_3', 'get_action1_slide_count_3', 'get_action2_slide_count_3', 'get_action3_slide_count_3', 'get_action4_slide_count_3', 'get_action5_slide_count_3', 'get_action0_slide_pct_3', 'get_action1_slide_pct_3', 'get_action2_slide_pct_3', 'get_action3_slide_pct_3', 'get_action4_slide_pct_3', 'get_action5_slide_pct_3', 'get_action0_slide_count_5', 'get_action1_slide_count_5', 'get_action2_slide_count_5', 'get_action3_slide_count_5', 'get_action4_slide_count_5', 'get_action5_slide_count_5', 'get_action0_slide_pct_5', 'get_action1_slide_pct_5', 'get_action2_slide_pct_5', 'get_action3_slide_pct_5', 'get_action4_slide_pct_5', 'get_action5_slide_pct_5', 'get_action0_slide_count_7', 'get_action1_slide_count_7', 'get_action2_slide_count_7', 'get_action3_slide_count_7', 'get_action4_slide_count_7', 'get_action5_slide_count_7', 'get_action0_slide_pct_7', 'get_action1_slide_pct_7', 'get_action2_slide_pct_7', 'get_action3_slide_pct_7', 'get_action4_slide_pct_7', 'get_action5_slide_pct_7', 'get_action0_slide_count_10', 'get_action1_slide_count_10', 'get_action2_slide_count_10', 'get_action3_slide_count_10', 'get_action4_slide_count_10', 'get_action5_slide_count_10', 'get_action0_slide_pct_10', 'get_action1_slide_pct_10', 'get_action2_slide_pct_10', 'get_action3_slide_pct_10', 'get_action4_slide_pct_10', 'get_action5_slide_pct_10', 'repeat_video_count_max_all', 'repeat_video_count_skew_all', 'repeat_video_count_min_all', 'repeat_video_count_kurt_all', 'repeat_video_count_mean_all', 'repeat_video_count_std_all', 'watch_video_count_all', 'repeat_video_count_max_1', 'repeat_video_count_std_1', 'repeat_video_count_mean_1', 'repeat_video_count_min_1', 'repeat_video_count_skew_1', 'repeat_video_count_kurt_1', 'watch_video_count_1', 'repeat_video_count_mean_2', 'repeat_video_count_std_2', 'repeat_video_count_skew_2', 'repeat_video_count_max_2', 'repeat_video_count_kurt_2', 'repeat_video_count_min_2', 'watch_video_count_2', 'repeat_video_count_std_3', 'repeat_video_count_skew_3', 'repeat_video_count_mean_3', 'repeat_video_count_kurt_3', 'repeat_video_count_max_3', 'repeat_video_count_min_3', 'watch_video_count_3', 'repeat_video_count_mean_5', 'repeat_video_count_std_5', 'repeat_video_count_max_5', 'repeat_video_count_min_5', 'repeat_video_count_skew_5', 'repeat_video_count_kurt_5', 'watch_video_count_5', 'repeat_video_count_min_7', 'repeat_video_count_std_7', 'repeat_video_count_kurt_7', 'repeat_video_count_skew_7', 'repeat_video_count_mean_7', 'repeat_video_count_max_7', 'watch_video_count_7', 'repeat_video_count_kurt_10', 'repeat_video_count_std_10', 'repeat_video_count_max_10', 'repeat_video_count_min_10', 'repeat_video_count_skew_10', 'repeat_video_count_mean_10', 'watch_video_count_10', 'repeat_author_count_skew_all', 'repeat_author_count_kurt_all', 'repeat_author_count_mean_all', 'repeat_author_count_std_all', 'repeat_author_count_max_all', 'repeat_author_count_min_all', 'watch_author_count_all', 'repeat_author_count_kurt_1', 'repeat_author_count_min_1', 'repeat_author_count_max_1', 'repeat_author_count_skew_1', 'repeat_author_count_mean_1', 'repeat_author_count_std_1', 'watch_author_count_1', 'repeat_author_count_kurt_2', 'repeat_author_count_min_2', 'repeat_author_count_skew_2', 'repeat_author_count_max_2', 'repeat_author_count_mean_2', 'repeat_author_count_std_2', 'watch_author_count_2', 'repeat_author_count_mean_3', 'repeat_author_count_skew_3', 'repeat_author_count_std_3', 'repeat_author_count_kurt_3', 'repeat_author_count_max_3', 'repeat_author_count_min_3', 'watch_author_count_3', 'repeat_author_count_std_5', 'repeat_author_count_min_5', 'repeat_author_count_kurt_5', 'repeat_author_count_skew_5', 'repeat_author_count_mean_5', 'repeat_author_count_max_5', 'watch_author_count_5', 'repeat_author_count_max_7', 'repeat_author_count_min_7', 'repeat_author_count_skew_7', 'repeat_author_count_mean_7', 'repeat_author_count_std_7', 'repeat_author_count_kurt_7', 'watch_author_count_7', 'repeat_author_count_kurt_10', 'repeat_author_count_min_10', 'repeat_author_count_skew_10', 'repeat_author_count_mean_10', 'repeat_author_count_std_10', 'repeat_author_count_max_10', 'watch_author_count_10', 'activity_day_max', 'activity_day_mean', 'activity_day_kurt', 'activity_day_skew', 'activity_day_min', 'activity_day_std', 'activity_rest_day', 'activity_day_mode', 'most_activity_day', 'most_activity_day_sub_end_day', 'activity_day_diff_kurt', 'activity_day_diff_std', 'activity_day_diff_max', 'activity_day_diff_skew', 'activity_day_diff_min', 'activity_day_diff_mean', 'activity_day_diff_last', 'activity_day_unique_mean', 'activity_day_unique_std', 'activity_day_unique_skew', 'activity_day_unique_kurt', 'activity_day_unique_diff_mean', 'activity_day_unique_diff_kurt', 'activity_day_unique_diff_std', 'activity_day_unique_diff_max', 'activity_day_unique_diff_min', 'activity_day_unique_diff_skew', 'activity_day_unique_diff_last', 'activity_everyday_count_mean', 'activity_everyday_count_max', 'activity_everyday_count_kurt', 'activity_everyday_count_min', 'activity_everyday_count_std', 'activity_everyday_count_skew', 'activity_everyday_count_last', 'page0_everyday_count_max', 'page0_everyday_count_kurt', 'page0_everyday_count_min', 'page0_everyday_count_skew', 'page0_everyday_count_std', 'page0_everyday_count_mean', 'page0_everyday_count_last', 'page1_everyday_count_skew', 'page1_everyday_count_min', 'page1_everyday_count_kurt', 'page1_everyday_count_mean', 'page1_everyday_count_max', 'page1_everyday_count_std', 'page1_everyday_count_last', 'page2_everyday_count_skew', 'page2_everyday_count_max', 'page2_everyday_count_std', 'page2_everyday_count_min', 'page2_everyday_count_mean', 'page2_everyday_count_kurt', 'page2_everyday_count_last', 'page3_everyday_count_mean', 'page3_everyday_count_kurt', 'page3_everyday_count_skew', 'page3_everyday_count_max', 'page3_everyday_count_std', 'page3_everyday_count_min', 'page3_everyday_count_last', 'page4_everyday_count_max', 'page4_everyday_count_min', 'page4_everyday_count_mean', 'page4_everyday_count_std', 'page4_everyday_count_skew', 'page4_everyday_count_kurt', 'page4_everyday_count_last', 'action0_everyday_count_max', 'action0_everyday_count_std', 'action0_everyday_count_skew', 'action0_everyday_count_mean', 'action0_everyday_count_min', 'action0_everyday_count_kurt', 'action0_everyday_count_last', 'action1_everyday_count_max', 'action1_everyday_count_mean', 'action1_everyday_count_kurt', 'action1_everyday_count_std', 'action1_everyday_count_skew', 'action1_everyday_count_min', 'action1_everyday_count_last', 'action2_everyday_count_std', 'action2_everyday_count_kurt', 'action2_everyday_count_min', 'action2_everyday_count_max', 'action2_everyday_count_skew', 'action2_everyday_count_mean', 'action2_everyday_count_last', 'action3_everyday_count_max', 'action3_everyday_count_std', 'action3_everyday_count_skew', 'action3_everyday_count_min', 'action3_everyday_count_kurt', 'action3_everyday_count_mean', 'action3_everyday_count_last', 'action4_everyday_count_min', 'action4_everyday_count_kurt', 'action4_everyday_count_max', 'action4_everyday_count_std', 'action4_everyday_count_mean', 'action4_everyday_count_skew', 'action4_everyday_count_last', 'action5_everyday_count_skew', 'action5_everyday_count_min', 'action5_everyday_count_max', 'action5_everyday_count_mean', 'action5_everyday_count_kurt', 'action5_everyday_count_std', 'action5_everyday_count_last', 'last_launch_sub_register', 'last_video_sub_register', 'last_activity_sub_register', 'mean_launch_slide_count_all', 'mean_video_slide_count_all', 'mean_activity_slide_count_all', 'mean_page0_slide_count_all', 'mean_page1_slide_count_all', 'mean_page2_slide_count_all', 'mean_page3_slide_count_all', 'mean_page4_slide_count_all', 'mean_action0_slide_count_all', 'mean_action1_slide_count_all', 'mean_action2_slide_count_all', 'mean_action3_slide_count_all', 'mean_action4_slide_count_all', 'mean_action5_slide_count_all', 'mean_get_activity_slide_count_all', 'mean_get_page0_slide_count_all', 'mean_get_page1_slide_count_all', 'mean_get_page2_slide_count_all', 'mean_get_page3_slide_count_all', 'mean_get_page4_slide_count_all', 'mean_get_action0_slide_count_all', 'mean_get_action1_slide_count_all', 'mean_get_action2_slide_count_all', 'mean_get_action3_slide_count_all', 'mean_get_action4_slide_count_all', 'mean_get_action5_slide_count_all', 'launch_only_once']
        feature_df = pd.DataFrame({'Feature_name':feature_list})
        importance_list = [60, 0, 518, 683, 76, 1, 70, 47, 54, 68, 88, 78, 93, 31, 81, 33, 110, 71, 12, 72, 22, 23, 6, 22, 32, 23, 13, 2, 27, 15, 5, 5, 5, 4, 6, 5, 3, 2, 6, 8, 6, 5, 1, 6, 5, 13, 2, 0, 4, 4, 3, 7, 3, 1, 4, 0, 1, 2, 0, 2, 0, 1, 1, 0, 6, 3, 2, 5, 2, 15, 16, 19, 10, 11, 6, 19, 38, 37, 36, 17, 49, 25, 60, 21, 9, 7, 18, 27, 17, 15, 24, 16, 12, 16, 22, 14, 4, 21, 14, 14, 15, 7, 18, 28, 27, 7, 13, 2, 10, 28, 4, 15, 3, 19, 11, 25, 19, 15, 8, 7, 11, 20, 14, 16, 17, 20, 14, 17, 13, 6, 12, 18, 10, 15, 19, 32, 8, 18, 18, 6, 2, 12, 39, 21, 44, 15, 40, 9, 5, 1, 0, 0, 22, 46, 19, 17, 0, 3, 73, 7, 13, 2, 0, 0, 26, 8, 15, 7, 0, 0, 22, 4, 4, 2, 0, 0, 33, 23, 14, 8, 0, 5, 19, 12, 3, 6, 0, 0, 21, 4, 5, 9, 3, 0, 29, 13, 7, 4, 0, 0, 18, 19, 9, 12, 0, 0, 25, 7, 11, 7, 0, 0, 6, 11, 17, 3, 0, 2, 15, 7, 8, 9, 0, 0, 11, 45, 14, 21, 0, 4, 4, 0, 5, 7, 6, 2, 12, 9, 7, 13, 9, 2, 0, 3, 1, 2, 0, 0, 2, 7, 1, 1, 1, 0, 1, 1, 1, 2, 0, 6, 2, 5, 1, 4, 0, 0, 1, 1, 2, 0, 8, 4, 4, 7, 1, 0, 3, 4, 2, 10, 0, 4, 1, 4, 8, 3, 1, 1, 0, 1, 3, 2, 13, 3, 9, 7, 4, 0, 6, 3, 2, 3, 0, 4, 8, 3, 2, 4, 1, 0, 2, 0, 0, 4, 4, 0, 11, 0, 0, 1, 0, 0, 0, 0, 0, 4, 6, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 3, 4, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 5, 2, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 2, 5, 0, 2, 0, 0, 2, 1, 0, 4, 0, 0, 5, 4, 0, 5, 0, 0, 2, 0, 0, 7, 0, 0, 2, 4, 0, 2, 0, 0, 15, 23, 2, 21, 26, 25, 23, 9, 19, 15, 7, 19, 16, 23, 15, 19, 28, 5, 14, 2, 13, 18, 17, 10, 11, 5, 0, 14, 8, 13, 5, 4, 12, 12, 23, 4, 26, 6, 14, 8, 3, 18, 13, 16, 5, 8, 17, 16, 13, 13, 20, 29, 19, 9, 9, 28, 46, 6, 7, 26, 18, 15, 37, 16, 4, 14, 16, 18, 21, 14, 12, 17, 11, 14, 6, 4, 18, 8, 3, 13, 8, 9, 7, 18, 15, 3, 8, 18, 7, 21, 16, 9, 5, 19, 12, 13, 6, 13, 34, 67, 41, 21, 34, 46, 47, 11, 30, 7, 15, 37, 23, 11, 0, 14, 2, 30, 27, 23, 31, 26, 16, 32, 4, 9, 12, 5, 5, 14, 22, 5, 19, 14, 23, 16, 19, 17, 17, 23, 16, 24, 20, 15, 9, 19, 12, 25, 31, 19, 5, 24, 17, 13, 16, 23, 12, 30, 27, 13, 14, 17, 29, 6, 9, 6, 10, 6, 7, 22, 6, 12, 9, 5, 12, 7, 23, 6, 31, 14, 28, 16, 9, 16, 16, 12, 5, 9, 15, 12, 15, 5, 19, 5, 2, 4, 5, 9, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 8, 161, 9, 17, 167, 20, 70, 41, 69, 21, 11, 31, 55, 34, 14, 27, 3, 0, 9, 3, 7, 6, 5, 8, 6, 1, 0, 4, 0, 0, 0]
        importance_df = pd.DataFrame({'importance':importance_list})
        R_value_list = [0.08525993336004542, 0.08525993336004542, -0.09230017804053621, -0.05475007573475284, -0.10190505289582716, -0.10588152649860325, 0.477138636248096, 0.5239331642490557, 0.5707376251410735, 0.5887470099545994, 0.5881026690190719, 0.5698716701940373, 0.5371363615801114, -0.24118813421381072, 0.45317160237981907, -0.007068820808057499, 0.3208728291381997, 0.40569539519380154, 0.13693455670673682, -0.45975960640045277, 0.24401529702679628, 0.29200748146113653, 0.14857234422514448, 0.220717054848731, 0.16475227366824985, 0.1883283153275238, 0.15999002635701828, 0.11937247740465971, 0.0865906565527791, 0.10240974020604439, 0.11161189775775945, 0.11892998072525927, 0.12160541615676303, 0.12286480952817842, 0.13372450567972738, 0.027979634546493742, 0.14611609909601023, 0.16541795971429796, 0.010402928391546613, 0.15670522952849572, 0.03182634433712059, 0.10677632463518934, 0.15731371686875864, 0.06310359200622509, 0.09705076150300979, 0.05970847851953076, 0.12533346056789038, 0.04764719335378659, 0.10445376993701251, 0.10762496101827287, 0.0854811868387137, 0.13502400740817141, 0.15727658024954086, -0.039754286757675665, 0.01587877975667795, 0.12533346056789038, 0.11663269398875276, 0.03913570315615922, 0.10052055417133093, 0.08709920310989348, 0.0732999649549328, 0.11168909427168114, 0.11555114008871292, 0.09484473737232645, 0.0844596189516975, 0.11412744351281473, 0.07719277089975926, 0.029853904451555828, 0.10549727236938916, 0.3184684739151484, 0.2967699109198875, 0.23014520747264772, 0.19853533649035732, 0.23521282970478113, 0.07973196844927175, 0.1996714656343941, 0.11809406926328228, 0.03514344099580114, 0.13139349641826517, 0.0010728378874741917, 0.29096137712621223, 0.2611443141638049, 0.21313148968663484, 0.15823097260332267, 0.19749037630661911, 0.07538338020667, 0.34853315210736324, 0.24600767037448693, 0.17489477786462118, 0.2995709599614699, 0.0660722585684189, 0.3120951592645115, 0.2897891076736685, 0.23804214757914618, 0.1822944936565579, 0.20773640314465552, 0.0823431850053701, 0.36774829766943323, 0.24918780940134055, 0.1756098301801134, 0.31349585860933843, 0.06183478423604862, 0.3301669761681005, 0.30389370695335627, 0.24719863557760124, 0.19401744042276764, 0.23081152784869297, 0.08803439194042656, 0.3632740044982238, 0.24019340974300296, 0.16496963189513408, 0.3066544063238643, 0.05767038350378946, 0.34159183756145234, 0.3155916776225928, 0.25052061127352254, 0.20293599881225735, 0.2437496679877898, 0.08341534412034138, 0.34190611071561167, 0.21712548268850895, 0.14177332272153406, 0.2833154799179029, 0.046336485792979665, 0.340585987718811, 0.3158752182212436, 0.2478995024558539, 0.20388299437140753, 0.24450599497060954, 0.08410826191902249, 0.31933342627399675, 0.19590454016791703, 0.11954437379749919, 0.25869962010772707, 0.03522756452872474, 0.3342655500095971, 0.31072865377777936, 0.2418891559089519, 0.2039471007027617, 0.24334207359784196, 0.0843705657832177, 0.27438394072223127, 0.1620940697432621, 0.08618478112982876, 0.2102731460827253, 0.02230499206920429, 0.3168647920987287, 0.12951296044611374, 0.16667606663955226, 0.09894571373993628, 0.011535935307914759, 0.015471592110182715, 0.37417242438055065, -0.02742770812414117, 0.033809157205782564, 0.013079920154321314, -0.0017781035880911763, 0.001530071172979208, 0.28934395388553275, 0.1183209165479258, 0.12414080258964034, 0.048593031425332237, 0.008127458881072957, 0.009443064051787426, 0.5050112452902801, 0.1293058953587075, 0.11086705161043553, 0.059010870459855626, 0.0032775404427558335, 0.00824575470344988, 0.31030283512670254, 0.12637699936376004, 0.14040160516122083, 0.06888230014246759, 0.00943019855138108, 0.012980713228605396, 0.5439462207724028, 0.11891274852685797, 0.11196460409244433, 0.06215390829116192, 0.0025360948852493673, 0.008958690643138126, 0.32857323513592873, 0.13691363838667225, 0.1477331952118654, 0.07990363646843222, 0.0105716233419715, 0.014973725347953876, 0.5528314197018841, 0.10246948800849466, 0.10773451792040212, 0.05991713568546754, 0.0021630453965486235, 0.008929990022788954, 0.3397884056182852, 0.14151782231308804, 0.15795930724148774, 0.09138729495491649, 0.012587661096331358, 0.011738847352001086, 0.5395360981379004, 0.07838021054759463, 0.09711336211068937, 0.04985361816351566, 0.001717735107890993, 0.007591520047924313, 0.3386521346411155, 0.13970997169862884, 0.1687693071417182, 0.09575820000294182, 0.011199560296922444, 0.013188937634502847, 0.5162772331122428, 0.058809101944791366, 0.0879874425273343, 0.04230431537145028, 0.001336100222971521, 0.007311834866380139, 0.33247158530548454, 0.1357160873343291, 0.16913875647209123, 0.09726213798357089, 0.012054537717191687, 0.014962675254982719, 0.46659160230317975, 0.023197819219530614, 0.06742703268204797, 0.031475268760015, -0.0006258056425042621, 0.004645510386327422, 0.04294864087535574, 0.007280956104218405, 0.08353769164248079, 0.054816191889993204, 0.015905838300114775, 0.07653050178668412, 0.00883952855105493, 0.1308570469159704, 0.046740506989082574, 0.08975134663595158, 0.10268077974404087, 0.022166957121136758, 0.006517137167274529, 0.06802985045180418, 0.03648678581761611, 0.008346349681363555, 0.06127795265586305, 0.013394190729366498, 0.11920711180112667, 0.039157121348735266, 0.0936499458788509, 0.09173692476976786, 0.01877617498393558, 0.006232857117615426, 0.07437903255273864, 0.04398063090418586, 0.006998859477981113, 0.07169263928960952, 0.015363688893639555, 0.13373909598450845, 0.04547812643915088, 0.10595122227289737, 0.10585454612322794, 0.025855170923811528, 0.006439121806237247, 0.07841979148510327, 0.04819778902055066, 0.009384527034975047, 0.07642895368596692, 0.01523818769801924, 0.14131936945917842, 0.048281378469391734, 0.11155934228367578, 0.1137095998848042, 0.031356910108900106, 0.00696432887391445, 0.08227128243001233, 0.05208284655540331, 0.011330834887531324, 0.07933708347147038, 0.013847402530951953, 0.1465826491182826, 0.05039810934778388, 0.11232596401048325, 0.11950797378658018, 0.03221156761872488, 0.006905201965233984, 0.08467930375850091, 0.04638267193257901, 0.011682573211984248, 0.07907747716068729, 0.01332878346242068, 0.14605555241696483, 0.05263487663429082, 0.10957262606516298, 0.11899694615156815, 0.036909144231256005, 0.007205905744823299, 0.08618229713363622, 0.051316610713381006, 0.013429893588966504, 0.07876202616416443, 0.01092825850780702, 0.14206798570620358, 0.0514661995285878, 0.10249524623805994, 0.11434055206872051, 0.04117956984268514, 0.057853280372710286, 0.006495055234044211, 0.039691556685077165, 0, 0.0012138969963431, 0.16041208699862014, 0.05968185363298588, 0.0008629064303352521, 0.03340850049850384, 0, 0.0012138969963430989, 0.021091919557788098, 0.03941162626406463, 0.0040213703753556275, 0.01153733771753534, 0, 0, 0.15886055469721216, 0.05623254218792748, 0.003482271721568798, 0.02908464657429353, 0, 0, 0.017857795954447205, 0.04822443675521817, 0.0036983954707549707, 0.01876244859875402, 0, 0, 0.17691360211667975, 0.06552653104866706, 0.003148385836261728, 0.033829385949718474, 0, 0, 0.024622434203550966, 0.05337363451940945, 0.004307911870450136, 0.02446209211100845, 0, 0, 0.18594446004269102, 0.06943248194107889, 0.0023770745120247614, 0.03552189231199583, 0, 0, 0.029902233031333006, 0.05760152517552749, 0.005696402027619409, 0.03067025164817443, 0, 0, 0.18980015627072463, 0.07264508515566089, 0.0017059644901509584, 0.03807712093982749, 0, 0, 0.030735931940592615, 0.059524660413799585, 0.005457869273677361, 0.03482873207473347, 0, 0, 0.18741371986366967, 0.07188746664415863, 0.0006660987010209638, 0.038043480806980445, 0, 0, 0.03527818027015721, 0.05885652301435121, 0.005666309654078375, 0.03558781039885928, 0, 0.0012138969963431, 0.17856391430598606, 0.06900796005968343, 0.0006302022611790323, 0.03772161012667999, 0, 0.0012138969963430971, 0.052217064270287376, 0.39524353452522576, 0.07205203815962093, 0.21636404619276098, 0.09051602606011411, 0.05733539810787137, 0.3206517930744942, 0.05615436038827434, 0.013556910331566254, 0.035802699370239846, 0.10946143959150398, 0.40095102621334444, 0.25990501225881857, 0.2956200144066866, 0.04272928627409044, 0.01164723655095825, 0.4478953387797819, 0.035403360774042034, 0.27665434452490967, 0.1300320413220361, 0.3232080215027365, 0.012760759068441907, 0.46578201458565055, 0.04401957166485559, 0.27750003157487074, 0.041861332574954155, 0.13108039623966877, 0.3367382600353605, 0.0546558154496739, 0.016837436779621767, 0.048200932283811115, 0.2112439323786619, 0.4713841512270839, 0.26797043022126577, 0.34660941478891777, 0.19130586321981902, 0.06139276388094732, 0.2579572756155927, 0.464172794330106, 0.210720790429377, 0.05142387691278372, 0.34518738196244697, 0.23950582205213533, 0.05654767692569185, 0.053607114229972996, 0.12142147174459782, 0.440914604867807, 0.14402485320603284, 0.33766027792301584, 0.41241333214686426, 0.25982158743476197, 0.058711764823821525, 0.1732340944950413, 0.1842242273290844, -0.0022466804333409664, 0.31917460105473344, 0.24553936834447507, 0.019786250876822502, 0.18342281958881096, 0.38208060085827583, 0.04319867760669559, 0.20033685703605217, 0.28527291713650227, 0.26769675375128765, 0.08859696537592415, 0.4263309358552834, 0.14710624123584462, 0.043294208940543824, 0.04165701532677744, 0.3138636345659421, 0.04546063139126707, 0.4459462552619924, 0.048250824587102774, 0.27783847855795984, 0.17680732872277516, 0.07629591354789492, 0.32793851066035806, 0.04896973753278804, 0.05728018953500928, 0.2841043524069263, 0.4577873060614606, 0.037292259705290876, 0.18685272084358484, 0.3390548064558406, 0.17884984230520876, 0.048455416215290155, 0.4569003576968551, 0.13457507050576697, 0.19282683821547847, 0.28080443419602397, 0.3390955709760488, 0.2744538146552619, 0.0255578989440629, 0.44401720026001595, 0.10184443512314711, 0.18141729473825915, 0.1818454499283913, 0.3335686828273712, 0.498926147793087, 0.40960283507068496, 0.008822372729335514, 0.001575624550949769, 0.2845841386470865, 0.4258911819597157, -0.3032987644294152, 0.2893548172418082, 0.39379328944345243, -0.028572631872222566, 0.30006600580329357, 0.13342177394424398, 0.28364707078444035, 0.4318849911019041, -0.0034789088509277213, 0.03251587959944175, 0.014741875728171004, 0.4162532926736935, 0.4174488966728642, 0.007194851200333968, -0.232479591854272, 0.23610115438257992, 0.20997073495654342, 0.19093547806529712, 0.28364707078444035, 0.19180123398944632, 0.28942600164839266, 0.20567975111506828, 0.3046478983193468, 0.3544634754107023, 0.11738601811649102, 0.10035631921426874, 0.33078866640225923, 0.2763552098786917, 0.24531364257966046, 0.3291077309268183, 0.0873940485333295, 0.1069773096431589, 0.24209537027522401, 0.30496312390202807, 0.28462967254891536, 0.09555088987136513, 0.2169963793699042, 0.11684825851769719, 0.09322263749350423, 0.23836680771757954, 0.2598639521512873, 0.2380740777066481, 0.11573301508938694, 0.1808624055335952, 0.22277542862485317, 0.20692554960414622, 0.09551308250603942, 0.1957707860069254, 0.0751298287166055, 0.09576185915460868, 0.2309598499378436, 0.10468343875883326, 0.24246052416165612, 0.2605476543965679, 0.24929809257564825, 0.08476585244499968, 0.09724614075388226, 0.10773055519556739, 0.08094569566891914, 0.10974825368685895, 0.09139119645183294, 0.08633744104527771, 0.03270654017994118, 0.06647593261173071, 0.35346953524689895, 0.33044511167935214, 0.27674533905340987, 0.30587804973632793, 0.10269225449861599, 0.11766656113562436, 0.06042037434484117, 0.1395162120906886, 0.11896343624567825, 0.08758266420550342, 0.13017769148613198, 0.19413360667016968, 0.053852376100965625, 0.08060257988658519, 0.13167020954119193, 0.06646936598383438, 0.10359944417852271, 0.1573813840528897, 0.16567068269285432, 0.15307059886226707, 0.08752438950553522, 0.0987639714903541, 0.06095116463472105, 0.07445105430697392, 0.10340284944101884, 0.02707864371461608, 0.11600940727641841, 0.05344171159800918, 0.01092943605545985, 0.0008012695640223538, 0.011136794376942516, 0.0030597925738879888, 0.0111734670843755, 0.003080411257108099, 0.0076278648025391365, 0.00524621521905364, 0.021625970112951556, 0.015950514730199167, 0.019703070878867492, 0.002900119857436928, 0.005565555694506403, 0.019891085681239293, 0.4024486500188116, 0.16129142874731708, 0.40638325687474636, 0.579126791721233, 0.10562171313964915, 0.3580070014356135, 0.3183638662276223, 0.2597712395199483, 0.20750859710673225, 0.25669090514866877, 0.08791198116248143, 0.3571728428380212, 0.13744603573466604, 0.1600200540503888, 0.09066109501433788, 0.00981074099337287, 0.016655531978287428, 0.06811769587916548, 0.009572235562853828, 0.08697264295632079, 0.052872952502935736, 0.02656951882394269, 0.08912949796962433, 0.06576929715731782, 0.05982458322334742, 0.007320804850233004, 0.03912575843875814, 0, 0.001213896996343099, -0.32739465412999214]
        R_value_df = pd.DataFrame({'R_value':R_value_list})
        P_value_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.827655357733099e-09, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5670863513463235e-117, 0.0, 0.0, 1.0565762800436503e-17, 0.0, 1.6848904361820433e-151, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.7739636926187866e-235, 4.4067259693080275e-39, 0.0, 0.0, 4.449306608312054e-228, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.546396348771691e-133, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.689176653530062e-184, 0.0, 0.37694437052883567, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.79554e-319, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.594476881742607e-185, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.215519566500169e-75, 0.0, 0.0, 0.0, 0.0, 2.0831609257453124e-21, 3.434065548076226e-37, 0.0, 5.1411395575663306e-113, 1.0291373124300775e-170, 4.643036253339996e-27, 0.14309402366160137, 0.2076342787701707, 0.0, 0.0, 0.0, 0.0, 2.178751676085063e-11, 7.42362562547776e-15, 0.0, 0.0, 0.0, 0.0, 0.006949816095873001, 1.1140428772309811e-11, 0.0, 0.0, 0.0, 0.0, 8.071503458402406e-15, 1.1243228177082968e-26, 0.0, 0.0, 0.0, 0.0, 0.03674240090031976, 1.6054598540854838e-13, 0.0, 0.0, 0.0, 0.0, 3.132694947853574e-18, 6.063999428301022e-35, 0.0, 0.0, 0.0, 0.0, 0.07484922140605581, 1.9167567093392895e-13, 0.0, 0.0, 0.0, 0.0, 3.502352381548008e-25, 4.12720438565395e-22, 0.0, 0.0, 0.0, 0.0, 0.15717143882396384, 4.049227195624499e-10, 0.0, 0.0, 0.0, 0.0, 2.8693018106041556e-20, 1.743464451832157e-27, 0.0, 0.0, 0.0, 3.5145225119143016e-266, 0.2711790341265627, 1.7251704514608722e-09, 0.0, 0.0, 0.0, 0.0, 3.1473458912120075e-23, 6.789007434165754e-35, 0.0, 2.198232078458991e-81, 0.0, 3.218995383164171e-148, 0.6062840765717368, 0.0001303215375893697, 2.71948748620972e-274, 2.018125681030208e-09, 0.0, 0.0, 3.286146314454057e-39, 0.0, 3.338870738601497e-13, 0.0, 0.0, 0.0, 0.0, 1.789362340742406e-74, 7.993619595177045e-08, 0.0, 1.672740145193483e-198, 6.2517808773931344e-12, 0.0, 2.698245582185991e-28, 0.0, 2.5159752737171504e-228, 0.0, 0.0, 6.01824575196395e-54, 2.849395545569786e-07, 0.0, 1.5375016048129113e-287, 8.213824374300496e-09, 0.0, 1.0689477023508372e-36, 0.0, 2.5276527357407455e-307, 0.0, 0.0, 1.2182101472838262e-100, 1.1390200138091837e-07, 0.0, 0.0, 1.0853378129018165e-14, 0.0, 3.96497833018318e-36, 0.0, 0.0, 0.0, 0.0, 4.0336998903157754e-147, 9.718393550131989e-09, 0.0, 0.0, 1.040328985927947e-20, 0.0, 3.964453035352941e-30, 0.0, 0.0, 0.0, 0.0, 3.840943502049109e-155, 1.2938449262016109e-08, 0.0, 1.35087e-319, 6.483996957422388e-22, 0.0, 4.905057555309768e-28, 0.0, 0.0, 0.0, 0.0, 4.43347437297271e-203, 2.946868467844564e-09, 0.0, 0.0, 1.9447788985119691e-28, 0.0, 2.2513907583223536e-19, 0.0, 0.0, 0.0, 0.0, 2.575464666146735e-252, 0.0, 8.83993590139524e-08, 1.5098307330585727e-234, 1.0, 0.3174507149844249, 0.0, 0.0, 0.4773009423075947, 9.738834869703661e-167, 1.0, 0.3174507149844249, 1.3382299270454906e-67, 2.807683627938176e-231, 0.0009268791668896704, 2.060178461357089e-21, 1.0, 1.0, 0.0, 0.0, 0.004132738288415499, 7.653696438825249e-127, 1.0, 1.0, 5.724538234819176e-49, 0.0, 0.002320316643382992, 7.172981952545544e-54, 1.0, 1.0, 0.0, 0.0, 0.009517797488409196, 6.463546833847513e-171, 1.0, 1.0, 1.893091145874598e-91, 0.0, 0.00038844048138060903, 2.7530944948559402e-90, 1.0, 1.0, 0.0, 0.0, 0.05027081019177323, 3.029229474987743e-188, 1.0, 1.0, 5.793100509540425e-134, 0.0, 2.7142971018737013e-06, 7.83670859460455e-141, 1.0, 1.0, 0.0, 0.0, 0.1600345939477874, 5.201322178481166e-216, 1.0, 1.0, 1.9890578634220548e-141, 0.0, 6.96076961782761e-06, 4.793899316174573e-181, 1.0, 1.0, 0.0, 0.0, 0.5833016524679893, 1.2421691589518066e-215, 1.0, 1.0, 1.0684819174213604e-185, 0.0, 3.0629838791471697e-06, 6.156020895141109e-189, 1.0, 0.3174507149844249, 0.0, 0.0, 0.6037567226608929, 4.950376476852996e-212, 1.0, 0.3174507149844249, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.024438092216397e-29, 3.34508051066282e-191, 0.0, 0.0, 0.0, 0.0, 1.6207786332680107e-271, 8.60129417491527e-22, 0.0, 5.278262857166371e-187, 0.0, 0.0, 0.0, 7.802010911402279e-26, 0.0, 4.794684457552583e-288, 0.0, 1.1253557692159806e-260, 0.0, 0.0, 0.0, 9.965717349807486e-44, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06427530491240302, 0.0, 0.0, 1.0426033034685215e-59, 0.0, 0.0, 1.79348526705946e-277, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0802661403453742e-278, 3.723873199763132e-258, 0.0, 4.341965478622008e-307, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.8165114430920657e-207, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2037597604704476e-98, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.7071788864384614e-13, 0.19442035182307377, 0.0, 0.0, 0.0, 0.0, 0.0, 1.749985889206371e-122, 0.0, 0.0, 0.0, 0.0, 0.004169057185752713, 4.746046015448479e-158, 6.373155995562762e-34, 0.0, 0.0, 3.1148784366103962e-09, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.924567315512122e-160, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.3176836472408916e-110, 0.0, 0.0, 2.231587915980204e-19, 0.5093250568424659, 4.6415909652242255e-20, 0.011738454852446649, 3.5055504403358825e-20, 0.011184157634230094, 3.34134367312453e-10, 1.556250957429886e-05, 5.678968004868687e-71, 2.0221743420477875e-39, 3.190746225304112e-59, 0.01692141200772172, 4.570989647281181e-06, 2.5292146324285185e-60, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.483518415095554e-16, 7.956601682831364e-43, 0.0, 3.185011819093333e-15, 0.0, 0.0, 3.586298851293325e-106, 0.0, 0.0, 0.0, 1.6481377799356496e-09, 5.796999891504644e-228, 1.0, 0.3174507149844249, 0.0]
        P_value_df = pd.DataFrame({'P_value':P_value_list})
        feature_to_select = pd.concat([feature_df,importance_df,R_value_df,P_value_df],axis=1)
        feature_selected = feature_to_select[((feature_to_select.P_value<=P_threshold) & ((feature_to_select.R_value>=R_threshold)|(feature_to_select.R_value<=-R_threshold)))&(feature_to_select.importance>=importance_threshold)]
        used_feature = feature_selected.Feature_name.tolist()
        train_feature = train[used_feature]
        train_label = train['label']
        online_test_feature = test[used_feature]
        print('importance_threshold:'+str(importance_threshold)+' R_threshold:'+str(R_threshold)+' P_threshold:'+str(P_threshold)+' 特征数：'+ str(train_feature.columns.size))
    train_feature, offline_test_feature, train_label, offline_test_label = train_test_split(train_feature, train_label, test_size=0.1,random_state=624)
    return train_feature,train_label,online_test_feature,test_userid,offline_test_feature,offline_test_label

def auc_score(params):
    cv_auc = []
    skf = StratifiedKFold(n_splits=N,shuffle=False,random_state=624)
    for train_in,test_in in skf.split(train_feature,train_label):
        if type(train_feature)==pd.core.frame.DataFrame:
            X_train,X_test,y_train,y_test = train_feature.iloc[train_in],train_feature.iloc[test_in],train_label.iloc[train_in],train_label.iloc[test_in]
        elif type(train_feature)==np.ndarray:
            X_train,X_test,y_train,y_test = train_feature[train_in],train_feature[test_in],train_label[train_in],train_label[test_in]
    
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=10000,
                        valid_sets=lgb_eval,
                        verbose_eval=False,
                        early_stopping_rounds=20)
    
        y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        cv_auc.append(roc_auc_score(y_test,y_pred))
    mean_cv_auc = np.sum(cv_auc)/5
    return mean_cv_auc
        
def param_tune():
    print('Tuning params...')
    params = {
              'boosting_type': 'gbdt',
              'objective': 'binary',
              'metric': 'auc',
              }
    max_auc = auc_score(params)
    print('best auc updated:',max_auc)
    best_params = {}
    
    # print("调参1：学习率")
    # for learning_rate in [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]: 
    #     print('============================',learning_rate)
    #     params['learning_rate'] = learning_rate
    #     auc = auc_score(params)
    #     if auc > max_auc:
    #         max_auc = auc
    #         print('best auc updated:',max_auc)
    #         best_params['learning_rate'] = learning_rate
    # if 'learning_rate' in best_params:
    #     params['learning_rate'] = best_params['learning_rate']
    # else:
    #     del params['learning_rate']
    params['learning_rate'] = 0.05
    # print('调参1结束, best_params:',best_params)
    
    # print("调参2：提高准确率")
    # for max_depth in range(3,11,1):     
    #     for num_leaves in range(10,min(100,2**max_depth),10):#2**max_depth
    #         print('max_depth,num_leaves===========',max_depth,num_leaves)
    #         params['num_leaves'] = num_leaves
    #         params['max_depth'] = max_depth
    #         auc = auc_score(params)
    #         if auc > max_auc:
    #             max_auc = auc
    #             print('best auc updated:',max_auc)
    #             best_params['num_leaves'] = num_leaves
    #             best_params['max_depth'] = max_depth
    # if 'num_leaves' in best_params:
    #     params['num_leaves'] = best_params['num_leaves']
    #     params['max_depth'] = best_params['max_depth']
    # else:
    #     del params['num_leaves'],params['max_depth']
    # # # params['num_leaves'] = 220
    # # # params['max_depth'] = 9
    # print('调参2结束, best_params:',best_params)
    
    # print("调参3：降低过拟合")
    # for min_data_in_leaf in range(10,510,10): #(10,200,5)
    #     print('min_data_in_leaf=============',min_data_in_leaf)
    #     params['min_data_in_leaf'] = min_data_in_leaf
    #     auc = auc_score(params)
    #     if auc > max_auc:
    #         max_auc = auc
    #         print('best auc updated:',max_auc)
    #         best_params['min_data_in_leaf'] = min_data_in_leaf
    # if 'min_data_in_leaf' in best_params:
    #     params['min_data_in_leaf'] = best_params['min_data_in_leaf']
    # else:
    #     del params['min_data_in_leaf']
    # # # params['min_data_in_leaf'] = 210
    # print('调参3结束, best_params:',best_params)
    
    # print("调参4：采样")
    # for feature_fraction in [0.5,0.6,0.7,0.8,0.9,1.0]: #[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    #     for bagging_fraction in [0.5,0.6,0.7,0.8,0.9,1.0]:
    #         for bagging_freq in range(0,25,5): #
    #             print('feature_fraction,bagging_fraction,bagging_freq=====',feature_fraction,bagging_fraction,bagging_freq)
    #             params['feature_fraction'] = feature_fraction
    #             params['bagging_fraction'] = bagging_fraction
    #             params['bagging_freq'] = bagging_freq
    #             auc = auc_score(params)
    #             if auc > max_auc:
    #                 max_auc = auc
    #                 print('best auc updated:',max_auc)
    #                 best_params['feature_fraction'] = feature_fraction
    #                 best_params['bagging_fraction'] = bagging_fraction
    #                 best_params['bagging_freq'] = bagging_freq
    # if 'feature_fraction' in best_params:
    #     params['feature_fraction'] = best_params['feature_fraction']
    #     params['bagging_fraction'] = best_params['bagging_fraction']
    #     params['bagging_freq'] = best_params['bagging_freq']
    # else:
    #     del params['feature_fraction'],params['bagging_fraction'],params['bagging_freq']
    # # # params['feature_fraction'] = 0.9
    # # # params['bagging_fraction'] = 0.8
    # # # params['bagging_freq'] = 5
    # print('调参4结束, best_params:',best_params)
    
    # print("调参5：正则化")
    # for lambda_l1 in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]: #[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    #     for lambda_l2 in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]: #[0,0.2,0.4,0.6,0.8,1.0]
    #         for min_split_gain in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    #             print('lambda_l1,lambda_l2,min_split_gain=====',lambda_l1,lambda_l2,min_split_gain)
    #             params['lambda_l1'] = lambda_l1
    #             params['lambda_l2'] = lambda_l2
    #             params['min_split_gain'] = min_split_gain
    #             auc = auc_score(params)
    #             if auc > max_auc:
    #                 max_auc = auc
    #                 print('best auc updated:',max_auc)
    #                 best_params['lambda_l1'] = lambda_l1
    #                 best_params['lambda_l2'] = lambda_l2
    #                 best_params['min_split_gain'] = min_split_gain
    # if 'lambda_l1' in best_params:
    #     params['lambda_l1'] = best_params['lambda_l1']
    #     params['lambda_l2'] = best_params['lambda_l2']
    #     params['min_split_gain'] = best_params['min_split_gain']
    # else:
    #     del params['lambda_l1'],params['lambda_l2'],params['min_split_gain']
    # # # params['lambda_l1'] = 1
    # # # params['lambda_l2'] = 1
    # # # params['min_split_gain'] = 1
    # print('调参5结束, best_params:',best_params)

    return params

def train_predict_func_cv(params):
    cv_auc = []
    offline_auc = []
    cv_prediction = []
    model_i = 0
    model_path = '/home/kesci/Kesci-file/Model/'
    print('All params:',params)
    skf = StratifiedKFold(n_splits=N,shuffle=False,random_state=624)
    for train_in,test_in in skf.split(train_feature,train_label):
        if type(train_feature)==pd.core.frame.DataFrame:
            X_train,X_test,y_train,y_test = train_feature.iloc[train_in],train_feature.iloc[test_in],train_label.iloc[train_in],train_label.iloc[test_in]
        elif type(train_feature)==np.ndarray:
            X_train,X_test,y_train,y_test = train_feature[train_in],train_feature[test_in],train_label[train_in],train_label[test_in]
    
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=10000,
                        valid_sets=lgb_eval,
                        verbose_eval=False,
                        early_stopping_rounds=20)
        
        if is_save_model:
            gbm.save_model(model_path+'model_'+str(model_i)+'.txt')
            model_i += 1
    
        y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        cv_auc.append(roc_auc_score(y_test,y_pred))
        offline_y_pred = gbm.predict(offline_test_feature, num_iteration=gbm.best_iteration)
        offline_auc.append(roc_auc_score(offline_test_label,offline_y_pred))
        cv_prediction.append(gbm.predict(online_test_feature, num_iteration=gbm.best_iteration))
    
    mean_cv_auc = np.sum(cv_auc)/N
    mean_offline_auc = np.sum(offline_auc)/N
    mean_cv_prediction = np.sum(cv_prediction,axis=0)/N
    print('mean_cv_auc:',mean_cv_auc)
    print('mean_offline_auc:',mean_offline_auc)
    
    if is_save_result:
        result = pd.DataFrame()
        result['userid'] = list(test_userid.values)
        result['probability'] = list(mean_cv_prediction)
        time_date = time.strftime('%m-%d-%H-%M',time.localtime(time.time()))
        # submit_file_name = '%s_%s.txt'%(str(time_date),str(mean_offline_auc).split('.')[1])
        submit_file_name = '%s_%s_%s.txt'%(str(time_date),('%.6f' % mean_cv_auc).split('.')[1],('%.6f' % mean_offline_auc).split('.')[1])
        result.to_csv(submit_file_name,index=False,sep=',',header=None)
        print(submit_file_name+' 线上:{}')
    
    if is_show_importance:
        print('所用特征：'+ str(list(train_feature.columns.values)))
        # gbm = lgb.Booster(model_file=model_path+'model_'+str(1)+'.txt')
        print('所用特征重要性：'+ str(list(gbm.feature_importance())))
        fig, ax = plt.subplots(1, 1, figsize=[16, 100])
        lgb.plot_importance(gbm, ax=ax, max_num_features=700)
        plt.savefig('feature_importance.png')


if __name__ == "__main__":
    time_start = datetime.now()
    print('Start time:',time_start.strftime('%Y-%m-%d %H:%M:%S'))
    
    train_feature,train_label,online_test_feature,test_userid,offline_test_feature,offline_test_label = \
    feature_selection(feature_mode=1,importance_threshold=1,R_threshold=0.01,P_threshold=0.05,train=train666,test=test666,test_userid=test_userid666) # 1:all 2:importance 3:pearson 4:importance+pearson
    
    N = 5
    is_show_importance = True
    is_save_result = True
    is_save_model = False
    param_mode = 2 # 1:tune 2:no tune
    train_predict_mode = 1 # 1:cv 2:no cv
    
    if param_mode == 1:
        params = param_tune()
    elif param_mode == 2:
        params = {'metric': 'auc', 'boosting_type': 'gbdt', 'learning_rate': 0.02, 'num_leaves': 32, 'max_depth': 5,'objective': 'binary', 'verbose': 1,'feature_fraction': 1, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'lambda_l1': 1, 'lambda_l2': 1}
        # params = {'verbose': 1, 'bagging_fraction': 0.8, 'num_leaves': 32, 'feature_fraction': 0.9, 'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc', 'lambda_l1': 1, 'learning_rate': 0.02, 'lambda_l2': 1, 'min_data_in_leaf': 20, 'bagging_freq': 5, 'max_depth': 5}

    if train_predict_mode == 1:
        train_predict_func_cv(params)
    elif train_predict_mode == 2:
        train_predict_func_no_cv(params)

    time_end = datetime.now()
    print('End time:',time_end.strftime('%Y-%m-%d %H:%M:%S'))
    print('Total time:',"%.2f" % ((time_end-time_start).seconds/60),'minutes')
    
    
