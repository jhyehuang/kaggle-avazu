import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
from sklearn.utils import check_random_state 
import pylab 
import sys
import time

import xgboost as xgb
from joblib import dump, load, Parallel, delayed

sys.path.append('D:/GitHub/jhye_tool')
from flags import FLAGS, unparsed
print(FLAGS.src_train_path)

#FLAGS, unparsed = parse_args()
#sys.path.append(FLAGS.tool_dir)
sys.path.append(FLAGS.tool_ml_dir)
from ml_utils import *
from data_preprocessing import one_line_data_preprocessing, \
    two_features_data_preprocessing,new_features_w,data_concat

import logging


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)


train_set_path = FLAGS.train_set_path
output = FLAGS.output_dir


# a1
#计算 特征中 1、不同用户出现的次数 2、不同设备id出现的次数 3、不同ip出现的次数 4、不同用户不同时间出现的次数
one_line_col=one_line_data_preprocessing(FLAGS.src_train_path,FLAGS.dst_app_path,FLAGS.dst_site_path)  

train = pd.read_csv(open(FLAGS.src_train_path, "r"))

train['one_day']=train['hour'] % 10000 / 100
train['one_day_hour'] = train['hour'] % 100
train['date_time'] = (train['one_day'] - 21) * 24 + train['one_day_hour']
train['day_hour_prev'] = train['date_time'] - 1
train['day_hour_next'] = train['date_time'] + 1

train['one_day']=train['one_day']
train['one_day_hour']=train['one_day_hour']
train['date_time']=train['date_time']
train['day_hour_prev']=train['day_hour_prev']
train['day_hour_next']=train['day_hour_next']

# a2
# 类别特征之间每俩个特征进行拼接 组成新特征
train,new_expvn=two_features_data_preprocessing(train)


#a3
#新特征 参考点击率 更新一个权重出来
new_features_w(train,new_expvn)

#a4
# data concat
data_concat(train,FLAGS.dst_app_path,FLAGS.dst_site_path)  


logging.debug("to count prev/current/next hour by ip ...")

logging.debug("to save train ...")

#dump(train, output + 'train.joblib_dat')


logging.debug("to generate traintv_mx .. ")
app_or_web = None
_start_day = 21
list_param = ['C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'banner_pos', 'device_type', 'device_conn_type']
feature_list_dict = {}

feature_list_name = 'tvexp3'
feature_list_dict[feature_list_name] = list_param + ['exp2' + vn for vn in new_expvn ]+ one_line_col

filter_tv = np.logical_and(train.one_day.values >= _start_day, train.one_day.values < 31)
filter_t1 = np.logical_and(train.one_day.values < 30, filter_tv)
filter_v1 = np.logical_and(~filter_t1, filter_tv)    
    
logging.debug(filter_tv.sum())


for vn in feature_list_dict[feature_list_name] :
    if vn not in train.columns:
        logging.debug("="*60 + vn)
        

traintv_mx = train.as_matrix(feature_list_dict[feature_list_name])

logging.debug(traintv_mx.shape)


logging.debug("to save traintv_mx ...")

traintv_mx_save = {}
traintv_mx_save['traintv_mx'] = traintv_mx
traintv_mx_save['click'] = train.click.values
traintv_mx_save['day'] = train.one_day.values
traintv_mx_save['site_id'] = train.site_id.values
dump(traintv_mx_save, FLAGS.tmp_data_path  +FLAGS.train_job_name+ '.joblib_dat')



