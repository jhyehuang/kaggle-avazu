import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
from sklearn.utils import check_random_state 
from sklearn.model_selection import train_test_split
import pylab 
import sys
sys.path.append('..')
import time
from joblib import dump, load, Parallel, delayed
import utils
from ml_utils import *

#sys.path.append(utils.xgb_path)
import xgboost as xgb


import logging


from flags import parse_args
FLAGS, unparsed = parse_args()

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)

train_save = load(FLAGS.tmp_data_path +'cat_features.csv')

y = train_save['click']
x = train_save.drop('click',axis=1)

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=4)

n_trees = FLAGS.xgb_n_trees


param = FLAGS.gbdt_param

np.random.seed(999)
train_idx = np.random.random_integers(0, 3, X_train.shape[0])
test_idx = np.random.random_integers(0, 3, X_test.shape[0])

predv_xgb = 0
ctr = 0
for idx in [0, 1, 2, 3]:
    filter1 = np.logical_and(X_train, np.logical_and(train_idx== idx , True))
    filter_v1 = np.logical_and(X_test, np.logical_and(test_idx== idx , True))
    xt1 = X_train[filter1, :]
    yt1 = y_train[filter1]
    if xt1.shape[0] <=0 or xt1.shape[0] != yt1.shape[0]:
        logging.debug(xt1.shape, yt1.shape)
        raise ValueError('wrong shape!')
    dtrain = xgb.DMatrix(xt1, label=yt1)
    dvalid = xgb.DMatrix(X_test[filter_v1], label=y_test[filter_v1])
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    logging.debug(xt1.shape, yt1.shape)

    plst = list(param.items()) + [('eval_metric', 'logloss')]
    xgb1 = xgb.train(plst, dtrain, n_trees, watchlist)
    #xgb_pred[rseed] = xgb1.predict(dtv3)
    #xgb_list[rseed] = xgb1
    
    ctr += 1
    predv_xgb += xgb1.predict(dvalid)
    logging.debug('-'*30, ctr, logloss(predv_xgb / ctr, y_test[filter_v1]))

logging.debug("to save validation predictions ...")
dump(predv_xgb / ctr, utils.tmp_data_path + 'xgb_pred_v.joblib_dat')

