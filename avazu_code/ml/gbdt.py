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
#    xgb_param=dict([(key,[params[key]]) for key in params])
    X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, y_train, train_size = 0.33,random_state = 0)
#    logging.debug(params)
    #直接调用xgboost，而非sklarn的wrapper类
    xgtrain = xgb.DMatrix(X_train_part, label = y_train_part)
    xgvalid = xgb.DMatrix(X_val, label=y_val)
#    boost = xgb.sklearn.XGBClassifier()
#    cvresult = GridSearchCV(boost,params, scoring='neg_log_loss',n_jobs=-1,cv=cv_folds)
#    cvresult.fit(X_train,y_train)
#    alg=cvresult.best_estimator_
    watchlist = [(xgvalid, 'eval'), (xgtrain, 'train')]
    
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], folds =cv_folds,
                      obj=logloss, early_stopping_rounds=early_stopping_rounds,)
#  
    cvresult.to_csv('1_nestimators.csv', index_label = 'n_estimators')
    #最佳参数n_estimators
    n_estimators = cvresult.shape[0]
    
    # 采用交叉验证得到的最佳参数n_estimators，训练模型
    alg.set_params(n_estimators = n_estimators)
    alg.fit(X_train_part,y_train_part,eval_metric=logloss)
    
#    print(n_estimators)
        
    #Predict training set:
    train_predprob = alg.predict_proba(X_val)[:,1]
    lloss = logloss(train_predprob,y_val)

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
#        n_estimators = [i for i in range(200,1000,1)]
        xgb1 = XGBClassifier(learning_rate =0.1,
         n_estimators=1000,
         max_depth=7,
         min_child_weight=1,
         gamma=0,
         subsample=0.8,
         colsample_bytree=0.8,
         objective='binary:logistic',
         nthread=-1,
         scale_pos_weight=1,
         seed=27)

        modelfit(xgb1, X_train,y_train, cv_folds = kfold)
        
        logging.debug("to save validation predictions ...")
        ret=dump(xgb1, FLAGS.tmp_data_path+'1-gdbt.model.joblib_dat') 
        logging.debug(ret)
    else:
        xgb1 = load(FLAGS.tmp_data_path+'1-gdbt.model.joblib_dat')
#        xgb1=pd.read_csv(FLAGS.tmp_data_path+'1-gdbt.csv')
#        dtest = xgb.DMatrix(X_test)
        xgb_pred = xgb1.predict(X_test)
        y_pred = [round(value,4) for value in xgb_pred]
        logging.debug('-'*30)
        logging.debug(y_pred)
        test_id=pd.read_csv(FLAGS.tmp_data_path+'test_id.csv')
        ret_pd = pd.concat([test_id, y_pred], axis = 1)
        ret_pd.to_csv(FLAGS.tmp_data_path+'1-gdbt.test.csv',index=False)
        
        
if __name__ == "__main__":
    done()
    done(False)
        

