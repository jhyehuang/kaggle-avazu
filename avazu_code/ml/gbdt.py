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
from data_preprocessing import *

#sys.path.append(utils.xgb_path)
import xgboost as xgb


import logging


from flags import FLAGS, unparsed


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)
#print(FLAGS)
train_save,test_save = gdbt_data_get(FLAGS.src_test_path)
print(train_save.shape)
y_train = train_save['click']
train_save.drop('click',axis=1,inplace=True)
X_train = train_save

test_save.drop('click',axis=1,inplace=True)
X_test=test_save

#X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=4)

n_trees = FLAGS.xgb_n_trees


param = FLAGS.gbdt_param

np.random.seed(999)
train_idx = np.random.random_integers(0, 3, X_train.shape[0])
test_idx = np.random.random_integers(0, 3, X_train.shape[0])
logging.debug(train_idx.shape)
predv_xgb = []
ctr = 0
for idx in [0, 1, 2, 3]:
    filter1 = np.logical_and(train_idx== idx , True)
    print(filter1)
    filter_v1 = np.logical_and(test_idx== idx , True)
    xt1 = X_train.loc[filter1, :]
    yt1 = y_train.loc[filter1]
    
    xt2 = X_train.loc[filter_v1, :]
    yt2 = y_train.loc[filter_v1]
    if xt1.shape[0] <=0 or xt1.shape[0] != yt1.shape[0]:
        logging.debug(xt1.shape, yt1.shape)
        raise ValueError('wrong shape!')
    dtrain = xgb.DMatrix(xt1, label=yt1)
    dvalid = xgb.DMatrix(xt2, label=yt2)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    logging.debug(xt1.shape)
    logging.debug(yt1.shape)

    plst = list(param.items()) + [('eval_metric', 'logloss')]
    xgb1 = xgb.train(plst, dtrain, n_trees, watchlist)
    xgb_pred = xgb1.predict(dvalid)
#    predv_xgb+=(xgb_pred)
    #xgb_list[rseed] = xgb1
    logging.debug(xgb_pred.shape)
    logging.debug(predv_xgb)
    logging.debug(ctr)

    ctr += 1
#    predv_xgb = xgb_pred+predv_xgb
#    logging.debug(predv_xgb.shape)
    logging.debug('-'*30)
    logging.debug( str(ctr))
    logging.debug(str(logloss(xgb_pred , y_train[filter_v1])))

logging.debug("to save validation predictions ...")
dump(predv_xgb , FLAGS.tmp_data_path + 'xgb_pred_v.joblib_dat')

