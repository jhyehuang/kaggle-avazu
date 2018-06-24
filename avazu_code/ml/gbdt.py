import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
from sklearn.utils import check_random_state 
from sklearn.model_selection import GridSearchCV,StratifiedKFold,train_test_split
from xgboost import XGBClassifier
import xgboost as xgb


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

n_trees = FLAGS.xgb_n_trees



param = FLAGS.gbdt_param

#直接调用xgboost内嵌的交叉验证（cv），可对连续的n_estimators参数进行快速交叉验证
#而GridSearchCV只能对有限个参数进行交叉验证
def modelfit(alg, X_train,y_train, cv_folds=None, early_stopping_rounds=10):
    xgb_param = alg.get_xgb_params()
    xgb_param['num_class'] = 2
    
    #直接调用xgboost，而非sklarn的wrapper类
    xgtrain = xgb.DMatrix(X_train, label = y_train)
        
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], folds =cv_folds,
             metrics='logloss', early_stopping_rounds=early_stopping_rounds,)
  
    cvresult.to_csv('1_nestimators.csv', index_label = 'n_estimators')
    
    #最佳参数n_estimators
    n_estimators = cvresult.shape[0]
    
    # 采用交叉验证得到的最佳参数n_estimators，训练模型
    alg.set_params(n_estimators = n_estimators)
    alg.fit(X_train,y_train,eval_metric='loglos')
    print(n_estimators)
        
    #Predict training set:
    train_predprob = alg.predict_proba(X_train)
    _,lloss = logloss(y_train, train_predprob)

   #Print model report:
    print ("logloss of train :" )
    print(lloss)

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
#        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        xgb1 = XGBClassifier(FLAGS.gbdt_param)

        modelfit(xgb1, X_train,y_train, cv_folds = kfold)
        
        logging.debug("to save validation predictions ...")
        ret=dump(xgb1, FLAGS.tmp_data_path+'1-gdbt.model.joblib_dat') 
        logging.debug(ret)
    else:
        xgb1 = load(FLAGS.tmp_data_path+'1-gdbt.model.joblib_dat')
#        xgb1=pd.read_csv(FLAGS.tmp_data_path+'1-gdbt.csv')
        dtest = xgb.DMatrix(X_test)
        xgb_pred = xgb1.predict(dtest)
        y_pred = [round(value) for value in xgb_pred]
        logging.debug('-'*30)
        logging.debug(y_pred)
        
if __name__ == "__main__":
    done()
    done(False)
        

