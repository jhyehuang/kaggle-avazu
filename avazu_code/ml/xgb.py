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

#gpu_dict={'gpu_id':0,'max_bin':16,'tree_method':['gpu_hist'],'predictor':['gpu_predictor']}
#gpu_dict={'gpu_id':0,'tree_method':'gpu_hist','updater':'grow_gpu'}
gpu_dict={'tree_method':'gpu_hist',}

#  valid values are: {'approx', 'auto', 'exact', 'gpu_exact', 'gpu_hist', 'hist'}"
#params=param.update(gpu_dict)
#直接调用xgboost内嵌的交叉验证（cv），可对连续的n_estimators参数进行快速交叉验证
#而GridSearchCV只能对有限个参数进行交叉验证



def modelfit_cv(alg, X_train, y_train,cv_folds=None, early_stopping_rounds=10,cv_type='n_estimators',random_state=0):
    X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, y_train, train_size = 0.9,random_state = random_state)
    if cv_type=='n_estimators':
        xgb_param = alg.get_xgb_params()
#        xgb_param['num_class'] = 2

        xgtrain = xgb.DMatrix(X_train, label = y_train)
        
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], folds =cv_folds,
                         metrics='logloss', early_stopping_rounds=early_stopping_rounds)
        
        n_estimators = cvresult.shape[0]
        alg.set_params(n_estimators = n_estimators)
        
        print (cvresult)

        cvresult.to_csv( FLAGS.tmp_data_path+'n_estimators.csv', index_label = 'n_estimators')
        
        # plot
        test_means = cvresult['test-logloss-mean']
        test_stds = cvresult['test-logloss-std'] 
        
        train_means = cvresult['train-logloss-mean']
        train_stds = cvresult['train-logloss-std'] 

        x_axis = range(0, n_estimators)
        pyplot.errorbar(x_axis, test_means, yerr=test_stds ,label='Test')
        pyplot.errorbar(x_axis, train_means, yerr=train_stds ,label='Train')
        pyplot.title("XGBoost n_estimators vs Log Loss")
        pyplot.xlabel( 'n_estimators' )
        pyplot.ylabel( 'Log Loss' )
        pyplot.savefig(  FLAGS.tmp_data_path+'n_estimators4_2_3_699.png' )
        
    elif cv_type=='max_depth':
#        xgb_param = alg.get_xgb_params()
        max_depth = range(6,9,1)
        min_child_weight = range(1,6,1)
        param_cv = dict(max_depth=max_depth, min_child_weight=min_child_weight)

        cvresult = GridSearchCV(alg,param_grid=param_cv, scoring='neg_log_loss',n_jobs=-1,pre_dispatch='n_jobs',cv=cv_folds)
        cvresult.fit(X_train,y_train)
        pd.DataFrame(cvresult.cv_results_).to_csv('my_preds_maxdepth_min_child_weights_1.csv')
    #  
        #最佳参数n_estimators
        logging.debug(cvresult.best_params_)
#        param_cv.update(cvresult.best_params_)
        for key,value in cvresult.best_params_.items():
            alg.set_params(key=value)
    elif cv_type=='min_child_weight':
#        xgb_param = alg.get_xgb_params()
        min_child_weight = range(1,6,1)
        param_cv = dict(min_child_weight=min_child_weight)

        cvresult = GridSearchCV(alg,param_grid=param_cv, scoring='neg_log_loss',n_jobs=-1,pre_dispatch=3,cv=cv_folds)
        cvresult.fit(X_train,y_train)
        pd.DataFrame(cvresult.cv_results_).to_csv('my_preds_maxdepth_min_child_weights_1.csv')
    #  
        #最佳参数n_estimators
        logging.debug(cvresult.best_params_)
#        param_cv.update(cvresult.best_params_)
        for key,value in cvresult.best_params_.items():
            alg.set_params(key=value)

    elif cv_type=='subsample':
        subsample = [i/10.0 for i in range(3,9)]
        colsample_bytree = [i/10.0 for i in range(6,10)]
        param_cv = dict(subsample=subsample, colsample_bytree=colsample_bytree)

        cvresult = GridSearchCV(alg,param_grid=param_cv, scoring='neg_log_loss',n_jobs=3,pre_dispatch='n_jobs',cv=cv_folds)
        cvresult.fit(X_train,y_train)
        pd.DataFrame(cvresult.cv_results_).to_csv('my_preds_subsampleh_colsample_bytree_1.csv')
    #  
        #最佳参数n_estimators
        logging.debug(cvresult.best_params_)
        for key,value in cvresult.best_params_.items():
            alg.set_params(key=value)
        
    elif cv_type=='reg_alpha':
        reg_alpha = [ 1.5, 2]    #default = 0, 测试0.1,1，1.5，2
        reg_lambda = [0.5, 1, 2]      #default = 1，测试0.1， 0.5， 1，2
        
        param_cv = dict(reg_alpha=reg_alpha, reg_lambda=reg_lambda)

        cvresult = GridSearchCV(alg,param_grid=param_cv, scoring='neg_log_loss',n_jobs=3,pre_dispatch='n_jobs',cv=cv_folds)
        cvresult.fit(X_train,y_train)
        pd.DataFrame(cvresult.cv_results_).to_csv('reg_alpha_vs_reg_lambda1.csv')
    #  
        #最佳参数n_estimators
        logging.debug(cvresult.best_params_)
        for key,value in cvresult.best_params_.items():
            alg.set_params(key=value)

    #Fit the algorithm on the data
    alg.fit(X_train, y_train, eval_metric='logloss')
        
    #Predict training set:
    
    train_predprob = alg.predict_proba(X_val)
    try:
        logloss = log_loss(y_val, train_predprob)
        logging.debug(logloss)
    except:
        pass
    
    try:
        _,lloss = logloss(train_predprob[:,1],y_val)

       #Print model report:
        logging.debug ("logloss of train :" )
        logging.debug(lloss)
    except:
        pass


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

def done(istrain=True):
#    test_save.drop('click',axis=1,inplace=True)
#    op=['n_estimators','max_depth','min_child_weight','subsample','reg_alpha']
    op=['fin']
    if istrain:
        train_save = gdbt_data_get_train(100)
        print(train_save.shape)
        y_train = train_save['click']
        train_save.drop('click',axis=1,inplace=True)
        X_train = train_save
#        dtrain = xgb.DMatrix(X_train, label=y_train)
#        n_estimators = [i for i in range(200,1000,1)]
        xgb1 = XGBClassifier(learning_rate =0.1,
        n_estimators=666,
        max_depth=6,
        min_child_weight=1,
#        gamma=0.1,
#        subsample=0.8,
#        colsample_bytree=0.8,
        scoring='roc_auc',
        objective='binary:logistic',
        eval_metric=['logloss','auc'],
        nthread=-1,
        verbose=2,
#        scale_pos_weight=1,
#        reg_alpha=1.5,
#        reg_lambda=0.5,
        seed=27,
        silent=0,**gpu_dict)
        for i,oper in enumerate(op):
            modelfit_cv(xgb1, X_train,y_train, cv_folds = kfold,cv_type=oper,random_state=i)        
            logging.debug(oper+":to save validation predictions ...")
            ret=dump(xgb1, FLAGS.tmp_data_path+'xgboost.cv_'+oper+'.model.joblib_dat') 
            logging.debug(ret)
        del train_save
        del X_train
        del y_train
    else:
        X_test = gdbt_data_get_test()
        print(X_test.shape)
#        X_test.drop('click',axis=1,inplace=True)

        for oper in op:
            xgb1 = load(FLAGS.tmp_data_path+'xgboost.cv_'+oper+'.model.joblib_dat')
            dtrain_predprob = xgb1.predict_proba(X_test)[:,1]
            logging.debug(dtrain_predprob)
            y_pred = [round(value,4) for value in dtrain_predprob]
            logging.debug('-'*30)
            y_pred=np.array(y_pred).reshape(-1,1)
            logging.debug(y_pred.shape)
            test_id=pd.read_csv(FLAGS.tmp_data_path+'test_id.csv')
            logging.debug(test_id['id'].shape)
            test_id['id']=test_id['id'].map(int)
            test_id['click']=y_pred
            test_id.to_csv(FLAGS.tmp_data_path+'1-'+oper+'-xgboost.test.csv',index=False)
        del X_test
        
        
if __name__ == "__main__":
    done()
    done(False)
        
