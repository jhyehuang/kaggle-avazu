import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
from sklearn.utils import check_random_state 
from sklearn.model_selection import GridSearchCV,StratifiedKFold,train_test_split
from xgboost import XGBClassifier
import xgboost as xgb
import lightgbm as lgbm

from sklearn.metrics import log_loss
 
import sys
sys.path.append('..')
import time
from joblib import dump, load, Parallel, delayed
import utils
from ml_utils import *
from data_preprocessing import *
from matplotlib import pyplot
import seaborn as sns


#sys.path.append(utils.xgb_path)
import xgboost as xgb


import logging


from flags import FLAGS, unparsed


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)

n_trees = FLAGS.xgb_n_trees



param = FLAGS.gbdt_param

#直接调用LightGBM内嵌的交叉验证（cv），可对连续的n_estimators参数进行快速交叉验证
#而GridSearchCV只能对有限个参数进行交叉验证
import json
def modelfit(params, alg, X_train, y_train, early_stopping_rounds=10):
    lgbm_params = params.copy()
    lgbm_params['num_class'] = 2
    
    #直接调用LightGBM，而非sklarn的wrapper类
    lgbmtrain = lgbm.Dataset(X_train, y_train, silent=True)
    
    cv_result = lgbm.cv(
        lgbm_params, lgbmtrain, num_boost_round=10000, nfold=5, stratified=False, shuffle=True, metrics='multi_logloss',
        early_stopping_rounds=early_stopping_rounds,show_stdv=True,seed=0)
    # note: cv_results will look like: {"multi_logloss-mean": <a list of historical mean>,
    # "multi_logloss-stdv": <a list of historical standard deviation>}
    print('best n_estimators:', len(cv_result['multi_logloss-mean']))
    print('best cv score:', cv_result['multi_logloss-mean'][-1])
    #cv_result.to_csv('lgbm1_nestimators.csv', index_label = 'n_estimators')
    json.dump(cv_result, open('lgbm_1.json', 'w'))
    
    # 采用交叉验证得到的最佳参数n_estimators，训练模型
    alg.set_params(n_estimators = len(cv_result['multi_logloss-mean']))
    alg.fit(X_train, y_train)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

def done(istrain=True):
    train_save,test_save = gdbt_data_get(FLAGS.src_test_path)
    print(train_save.shape)
    y_train = train_save['click']
    train_save.drop('click',axis=1,inplace=True)
    X_train = train_save
    
    test_save.drop('click',axis=1,inplace=True)
    X_test=test_save
    if istrain:
        params = {'boosting_type': 'gbdt', 
                  'objective': 'multiclass', 
                  'nthread': -1, 
                  'silent': True,
                  'learning_rate': 0.1, 
                  'num_leaves': 50, 
                  'max_depth': 6,
                  'max_bin': 127, 
                  'subsample_for_bin': 50000,
                  'subsample': 0.8, 
                  'subsample_freq': 1, 
                  'colsample_bytree': 0.8, 
                  'reg_alpha': 1, 
                  'reg_lambda': 0,
                  'min_split_gain': 0.0, 
                  'min_child_weight': 1, 
                  'min_child_samples': 20, 
                  'scale_pos_weight': 1}

        lgbm1 = lgbm.sklearn.LGBMClassifier(num_class= 9, n_estimators=1000, seed=0, **params)

        modelfit(params,lgbm1, X_train, y_train)
        
        logging.debug("to save validation predictions ...")
        ret=dump(lgbm1, FLAGS.tmp_data_path+'1-gdbt.model.joblib_dat') 
        logging.debug(ret)
    else:
        xgb1 = load(FLAGS.tmp_data_path+'1-gdbt.model.joblib_dat')
#        xgb1=pd.read_csv(FLAGS.tmp_data_path+'1-gdbt.csv')
#        dtest = xgb.DMatrix(X_test)
        xgb_pred = xgb1.predict(X_test)
        y_pred = [round(value,4) for value in xgb_pred]
        logging.debug('-'*30)
        logging.debug(y_pred)
        ret_list=X_test['id']
        ret_pd = pd.concat([ret_list, y_pred], axis = 1)
        ret_pd.to_csv(FLAGS.tmp_data_path+'1-gdbt.test.csv',index=False)
        
    
def plot_ret():
    #cv_result = pd.DataFrame.from_csv('lgbm1_nestimators.csv')
    cv_result = pd.read_json("lgbm_1.json")
    
    # plot
    test_means = cv_result['multi_logloss-mean']
    #test_stds = cv_result['multi_logloss-std'] 
    
    x_axis = range(0, cv_result.shape[0])
    pyplot.plot(x_axis, test_means) 
    
    pyplot.title("LightGBM n_estimators vs Log Loss")
    pyplot.xlabel( 'n_estimators' )
    pyplot.ylabel( 'Log Loss' )
    pyplot.savefig( 'lgbm1_n_estimators.png')
    
    pyplot.show()
    
if __name__ == "__main__":
#    done()
    done(False)
        

