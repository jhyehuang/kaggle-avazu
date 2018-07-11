import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
import pylab 
import sys
import time
import os
import utils
from utils import *
from joblib import dump, load, Parallel, delayed

sys.path.append(utils.xgb_path)
import xgboost as xgb

gpu_dict={'tree_method':'gpu_hist',}

import logging

from flags import FLAGS, unparsed

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)



def build_data():

    X_train = gdbt_data_get_train(100)
    print(X_train.shape)
    y_train = train_save['click']
    X_train.drop('click',axis=1,inplace=True)
    X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, y_train, train_size = 0.6,random_state = 7)
    
    dtrain = xgb.DMatrix(X_train_part, label=y_train_part)
    dvalid = xgb.DMatrix(X_val, label=y_val)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    print (xt1.shape, yt1.shape)

    param = {'max_depth':6, 'eta':.5, 'objective':'binary:logistic', 'verbose':0,
             'subsample':1.0, 'min_child_weight':50, 'gamma':0,
             'nthread': -1, 'colsample_bytree':.5, 'base_score':0.16, 'seed': 7}

    plst = list(param.items()) + [('eval_metric', 'logloss')]
    xgb_test_basis = xgb.train(plst, dtrain, n_trees, watchlist)

    dtv = xgb.DMatrix(X_train)
    xgb_leaves = xgb_test_basis.predict(dtv, pred_leaf = True)

    new_pd = pd.DataFrame({'click': y_train})
    print(xgb_leaves.shape)
    for i in range(n_trees):
        pred2 = xgb_leaves[:, i]
        print(i, np.unique(pred2).size)
        new_pd['xgb_basis'+str(i)] = pred2

    train_save = ffm_data_get_train(100)

    idx_base = 0
    for vn in ['xgb_basis' + str(i) for i in range(n_trees)]:
        _cat = np.asarray(new_pd[vn].astype('category').values.codes, dtype='int32')
        _cat1 = _cat + idx_base
        print(vn, idx_base, _cat1.min(), _cat1.max(), np.unique(_cat).size)
        train_save[vn] = _cat1
        idx_base += _cat.max() + 1

    train_save.to_csv(FLAGS.tmp_data_path+'xgb_new_features.csv',index=False)

build_data()
import gc
gc.collect()


