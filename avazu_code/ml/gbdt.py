import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
from sklearn.utils import check_random_state 
from sklearn.model_selection import GridSearchCV,StratifiedKFold,train_test_split
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import log_loss

from matplotlib import pyplot
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





FLAGS.gbdt_param
#gpu_dict={'gpu_id':0,'max_bin':16,'tree_method':'gpu_hist'}
max_depth = range(6,10,1)
min_child_weight = range(1,6,2)
params = dict(max_depth=max_depth, min_child_weight=min_child_weight)
#params=param.update(gpu_dict)
#直接调用xgboost内嵌的交叉验证（cv），可对连续的n_estimators参数进行快速交叉验证
#而GridSearchCV只能对有限个参数进行交叉验证


def modelfit_n_estimators(alg, X_train, y_train, useTrainCV=True, cv_folds=None, early_stopping_rounds=100):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
#        xgb_param['num_class'] = 9
        
        xgtrain = xgb.DMatrix(X_train, label = y_train)
        
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], folds =cv_folds,
                         metrics='logloss', early_stopping_rounds=early_stopping_rounds)
        
        n_estimators = cvresult.shape[0]
        alg.set_params(n_estimators = n_estimators)
        
        print (cvresult)

        cvresult.to_csv('my_preds.csv', index_label = 'n_estimators')
        
        # plot
        test_means = cvresult['test-mlogloss-mean']
        test_stds = cvresult['test-mlogloss-std'] 
        
        train_means = cvresult['train-mlogloss-mean']
        train_stds = cvresult['train-mlogloss-std'] 

        x_axis = range(0, n_estimators)
        pyplot.errorbar(x_axis, test_means, yerr=test_stds ,label='Test')
        pyplot.errorbar(x_axis, train_means, yerr=train_stds ,label='Train')
        pyplot.title("XGBoost n_estimators vs Log Loss")
        pyplot.xlabel( 'n_estimators' )
        pyplot.ylabel( 'Log Loss' )
        pyplot.savefig( 'n_estimators4_2_3_699.png' )
    
    #Fit the algorithm on the data
    alg.fit(X_train, y_train, eval_metric='logloss')
        
    #Predict training set:
    train_predprob = alg.predict_proba(X_train)
    logloss = log_loss(y_train, train_predprob)

        
    #Print model report:
    print ("logloss of train :" )
    print (logloss)



def modelfit_other(alg, X_train,y_train, cv_folds=None, early_stopping_rounds=10,is_cv=True):
    X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, y_train, train_size = 0.9,random_state = 0)
#    logging.debug(params)
    #直接调用xgboost，而非sklarn的wrapper类
    logging.debug(X_train_part.shape)
    logging.debug(y_train_part.shape)
    
    if  is_cv:
#        xgtrain = xgb.DMatrix(X_train_part, label = y_train_part)
#        xgvalid = xgb.DMatrix(X_val, label=y_val)
        logging.debug(X_val.shape)
        cvresult = GridSearchCV(alg,param_grid=params, scoring='neg_log_loss',n_jobs=-1,cv=cv_folds)
        cvresult.fit(X_train,y_train)
        pd.DataFrame(cvresult.cv_results_).to_csv('my_preds_maxdepth_min_child_weights_1.csv')
    #  
        #最佳参数n_estimators
        logging.debug(cvresult.best_params_)
        FLAGS.gbdt_param.update(cvresult.best_params_)
#        n_estimators = cvresult.shape[0]
    else:
#        n_estimators=FLAGS.xgb_n_trees
        pass
    
    # 采用交叉验证得到的最佳参数n_estimators，训练模型
    logging.debug(FLAGS.gbdt_param)
#    for key,value in FLAGS.gbdt_param.items():
#        alg.set_params(key=value)
    
    alg.fit(X_train_part,y_train_part,eval_metric='logloss')
    
#    print(n_estimators)
        
    #Predict training set:
    train_predprob = alg.predict_proba(X_val)[:,1]
    _,lloss = logloss(train_predprob,y_val)

   #Print model report:
    logging.debug ("logloss of train :" )
    logging.debug(lloss)

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
        xgb1 = XGBClassifier(learning_rate =0.02,
         n_estimators=699,
         max_depth=7,
         min_child_weight=1,
         gamma=0,
         subsample=0.8,
         colsample_bytree=0.8,
         objective='binary:logistic',
         nthread=-1,
         scale_pos_weight=1,
         seed=27,
         silent=0)

        modelfit_other(xgb1, X_train,y_train, cv_folds = kfold,is_cv=False)
#        modelfit_other(xgb1, X_train,y_train, cv_folds = kfold)
        
        logging.debug("to save validation predictions ...")
        ret=dump(xgb1, FLAGS.tmp_data_path+'1-gdbt.model.joblib_dat') 
        logging.debug(ret)
    else:
        xgb1 = load(FLAGS.tmp_data_path+'1-gdbt.model.joblib_dat')
#        xgb1=pd.read_csv(FLAGS.tmp_data_path+'1-gdbt.csv')
#        dtest = xgb.DMatrix(X_test)
        xgb_pred = xgb1.predict(X_test)
        dtrain_predprob = xgb1.predict_proba(X_test)[:,1]
        logging.debug(xgb_pred)
        logging.debug(dtrain_predprob)
        y_pred = [round(value,4) for value in dtrain_predprob]
        logging.debug('-'*30)
        y_pred=np.array(y_pred).reshape(-1,1)
        logging.debug(y_pred.shape)
        test_id=pd.read_csv(FLAGS.tmp_data_path+'test_id.csv')
        logging.debug(test_id['id'].shape)
        test_id['id']=test_id['id'].map(int)
        test_id['click']=y_pred
        test_id.to_csv(FLAGS.tmp_data_path+'1-gdbt.test.csv',index=False)
        
        
if __name__ == "__main__":
    done()
    done(False)
        

