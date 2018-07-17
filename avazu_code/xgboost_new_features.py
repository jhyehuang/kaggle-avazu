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

#sys.path.append(utils.xgb_path)
import xgboost as xgb
from data_preprocessing import *
gpu_dict={'tree_method':'gpu_hist',}

import logging

from flags import FLAGS, unparsed

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)


n_trees=666
def build_data(seed=100,is_type='train'):
    if 'train'==is_type:
        dtrain = xgb.DMatrix(FLAGS.tmp_data_path+'train'+str(seed)+'/xgboost.new_features.dtrain.joblib_dat')
        dvalid = xgb.DMatrix(FLAGS.tmp_data_path+'train'+str(seed)+'/xgboost.new_features.dvalid.joblib_dat')
        
    #    del y_train  
    
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    #    print (X_train_part.shape, y_train_part.shape)
    
        param = {'max_depth':6, 'eta':.1, 'objective':'binary:logistic', 'verbose':2,
                 'subsample':1.0, 'min_child_weight':1, 'gamma':0,
                 'nthread': -1, 'colsample_bytree':.8, 'base_score':0.16, 'seed': 27,'silent':0}
        param.update(gpu_dict)
        plst = list(param.items()) + [('eval_metric', 'logloss')]
        xgb_test_basis = xgb.train(plst, dtrain, n_trees, watchlist)
        del dtrain,dvalid
        gc.collect()
        dtv = xgb.DMatrix(FLAGS.tmp_data_path+'train'+str(seed)+'/xgboost.new_features.dtv.joblib_dat')
    #    dtv = xgb.DMatrix(X_train)
        xgb_leaves = xgb_test_basis.predict(dtv, pred_leaf = True)
    
        new_pd = pd.DataFrame()
        print(xgb_leaves.shape)
        for i in range(n_trees):
            pred2 = xgb_leaves[:, i]
            print(i, np.unique(pred2).size)
            new_pd['xgb_basis'+str(i)] = pred2
    
    #    train_save = gdbt_data_get_train(799)
    
        idx_base = 0
        for vn in ['xgb_basis' + str(i) for i in range(n_trees)]:
            _cat = np.asarray(new_pd[vn].astype('category').values.codes, dtype='int32')
            _cat1 = _cat + idx_base
            print(vn, idx_base, _cat1.min(), _cat1.max(), np.unique(_cat).size)
            new_pd[vn] = _cat1
            idx_base += _cat.max() + 1
        logging.debug(new_pd.shape)
        logging.debug(new_pd.head(3))
        new_pd.to_csv(FLAGS.tmp_data_path+'train'+str(seed)+'/xgb_new_features.csv',index=False)
        del new_pd,dtv
        gc.collect()
        xgb_test_basis.save_model(FLAGS.tmp_data_path+'xgb_new_features.model')
        
    elif 'test'==is_type:
        
        dtv = xgb.DMatrix(FLAGS.tmp_data_path+'test/xgboost.new_features.test.joblib_dat')
        logging.debug(dtv)
        xgb_test_basis = xgb.Booster({'nthread':-1}) #init model
        xgb_test_basis.load_model(FLAGS.tmp_data_path+'xgb_new_features.model') # load data
        xgb_leaves = xgb_test_basis.predict(dtv, pred_leaf = True)
    
        new_pd = pd.DataFrame()
        print(xgb_leaves.shape)
        for i in range(n_trees):
            pred2 = xgb_leaves[:, i]
            print(i, np.unique(pred2).size)
            new_pd['xgb_basis'+str(i)] = pred2
    
    #    train_save = gdbt_data_get_train(799)
    
        idx_base = 0
        for vn in ['xgb_basis' + str(i) for i in range(n_trees)]:
            _cat = np.asarray(new_pd[vn].astype('category').values.codes, dtype='int32')
            _cat1 = _cat + idx_base
            print(vn, idx_base, _cat1.min(), _cat1.max(), np.unique(_cat).size)
            new_pd[vn] = _cat1
            idx_base += _cat.max() + 1
        logging.debug(new_pd.shape)
        logging.debug(new_pd.head(3))
        new_pd.to_csv(FLAGS.tmp_data_path+'test/xgb_new_features.csv',index=False)



#build_data()

build_data(seed=25)

#build_data(is_type='test')



