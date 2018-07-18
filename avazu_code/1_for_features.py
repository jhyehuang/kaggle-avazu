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


sys.path.append(FLAGS.tool_ml_dir)
from ml.ml_utils import *
from data_preprocessing import *

import logging
import gc

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)


train_set_path = FLAGS.train_set_path
output = FLAGS.output_dir


#a2
# 将train 和test 拼到一起
#concat_train_test(FLAGS.src_train_path,FLAGS.src_test_path)

#gc.collect()


# a2
#计算 特征中 1、不同用户出现的次数 2、不同设备id出现的次数 3、不同ip出现的次数 4、不同用户不同时间出现的次数
#将训练集写入硬盘
#file1,file2,file3=one_line_data_preprocessing()  
#gc.collect()

#a4
#将点击数据按列写入硬盘
#click_to_csv()
#gc.collect()

#file1,file2,file3='cat_features.csv','date_list.csv','num_features.csv'
# a5
# 类别特征之间每俩个特征进行拼接 组成新特征
#new_expvn=two_features_data_preprocessing(FLAGS.tmp_data_path+file1,FLAGS.tmp_data_path+file2,FLAGS.tmp_data_path+file3)
#gc.collect()


# a6
# 按照时间维度对 特征进行概率计算
#features_by_chick()
#gc.collect()


# a7
# ouwenzhang
#ouwenzhang()
#gc.collect()

#a6
#新特征 参随机选择5个特征，计算先验概率
#new_features_w()
#gc.collect()


# 随机切割4个 500w的小数据集  调参
#get_train_split()
#gc.collect()


#切分训练集到25 中，测试集到test中
#get_train_test_split()

# 将训练数据集切分为  训练、验证、 转化为 Dmtrix
#gdbt_DM_get_train(25)

# 将测试集转化为 Dmtrix
#gdbt_DM_get_test()


# 训练集 one hot
train_data_ont_hot()
gc.collect()

# 验证集one hot
#vali_data_ont_hot(25)
#gc.collect()

# 测试集one hot
#test_data_ont_hot()
#gc.collect()

