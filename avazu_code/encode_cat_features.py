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

#sys.path.append('..')
from flags import FLAGS, unparsed
print(FLAGS.src_train_path)

#FLAGS, unparsed = parse_args()
#sys.path.append(FLAGS.tool_dir)
sys.path.append(FLAGS.tool_ml_dir)
from ml.ml_utils import *
from data_preprocessing import *

import logging


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)


train_set_path = FLAGS.train_set_path
output = FLAGS.output_dir


#a1
# 将train 和test 拼到一起
#train=concat_train_test(FLAGS.src_train_path,FLAGS.src_test_path)


#logging.debug(train.shape)
# a2
#计算 特征中 1、不同用户出现的次数 2、不同设备id出现的次数 3、不同ip出现的次数 4、不同用户不同时间出现的次数
#one_line_col=one_line_data_preprocessing(train,FLAGS.dst_app_path)  


#a3
#将训练集写入硬盘
#file1,file2,file3=data_to_col_csv(one_line_col,train,FLAGS.tmp_data_path)


#a4
#将点击数据按列写入硬盘
#click_to_csv()
#file1,file2,file3='cat_features.csv','date_list.csv','num_features.csv'


# a5
# 类别特征之间每俩个特征进行拼接 组成新特征
#new_expvn=two_features_data_preprocessing(FLAGS.tmp_data_path+file1,FLAGS.tmp_data_path+file2,FLAGS.tmp_data_path+file3)


#a6
#新特征 参随机选择5个特征，计算先验概率
#new_features_w()

# 随机切割4个 500w的小数据集  调参
#get_train_split()


#  将全部训练集 分割出来 
#get_train_test_split()


# one hot

train_data_ont_hot()

vali_data_ont_hot()

test_data_ont_hot()


