#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 23:12:53 2018

@author: zhijiehuang
"""

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
#cmd="grep ':cnt:' nohup.out"
#
#ret=os.popen(cmd)
#ret=ret.readlines()
#
#col_list=[]
#col_dict={}
#for x in ret:
##    print(x)
#    ls=x.split('-')[5].split(':')
#    if ls[2].replace('\n','')>'1':
##        print(ls[0],ls[2])
#        col_list.append(ls[0].replace(' ',''))
#        if int(ls[2].replace('\n',''))>100000:
#            col_dict[ls[0].replace(' ','')]=ls[2].replace('\n','')
#print(col_list)
#print(col_dict.keys())
#
#feature_score=pd.read_csv('/data/feature_score.csv')
#feature_score=feature_score[feature_score['importance']>100]
#print(list(feature_score['feature'].values))
def gdbt_data_get_test():
    test_save = pd.read_csv(FLAGS.tmp_data_path +'test/date_list.csv',)
    print (set(test_save.one_day.values))

gdbt_data_get_test()