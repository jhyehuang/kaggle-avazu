# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 19:55:38 2018

@author: admin
"""

import pandas as pd
import numpy as np

from sklearn.utils import check_random_state 
from sklearn.model_selection import GridSearchCV,StratifiedKFold,train_test_split
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
import lightgbm as lgb 

import logging


from flags import FLAGS, unparsed


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)

### 设置初始参数--不含交叉验证参数
print('设置参数')
cv_params = {
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric': 'binary_logloss',
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0
}


gpu_params = {
            'boosting_type': 'gbdt',
            'boosting': 'dart',
            'objective': 'binary',
            'metric': 'binary_logloss',
            
            'learning_rate': 0.01,
            'num_leaves':25,
            'max_depth':7,
            
            'max_bin':10,
            'min_data_in_leaf':8,
            
            'feature_fraction': 0.6,
            'bagging_fraction': 1,
            'bagging_freq':0,

            'lambda_l1': 0,
            'lambda_l2': 0,
            'min_split_gain': 0,
            'sparse_threshold': 1.0,
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0}

### 交叉验证(调参)


def modelfit_cv(lgb_train,cv_type='max_depth',):
    min_merror = float('Inf')
    logging.debug('交叉验证')
    best_params = {}
    if cv_type=='max_depth':
        # 准确率
        logging.debug("调参1：提高准确率")
        for num_leaves in range(20,200,5):
            for max_depth in range(3,8,1):
                cv_params['num_leaves'] = num_leaves
                cv_params['max_depth'] = max_depth
        
                cv_results = lgb.cv(
                                    cv_params,
                                    lgb_train,
                                    seed=2018,
                                    nfold=3,
                                    metrics=['binary_error'],
                                    early_stopping_rounds=10,
                                    verbose_eval=True
                                    )
                    
                mean_merror = pd.Series(cv_results['binary_error-mean']).min()
                boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()
                    
                if mean_merror < min_merror:
                    min_merror = mean_merror
                    best_params['num_leaves'] = num_leaves
                    best_params['max_depth'] = max_depth
                    
        cv_params['num_leaves'] = best_params['num_leaves']
        cv_params['max_depth'] = best_params['max_depth']
    elif cv_type=='max_bin':
        # 过拟合
        logging.debug("调参2：降低过拟合")
        for max_bin in range(1,255,5):
            for min_data_in_leaf in range(10,200,5):
                    cv_params['max_bin'] = max_bin
                    cv_params['min_data_in_leaf'] = min_data_in_leaf
                    
                    cv_results = lgb.cv(
                                        cv_params,
                                        lgb_train,
                                        seed=42,
                                        nfold=3,
                                        metrics=['binary_error'],
                                        early_stopping_rounds=3,
                                        verbose_eval=True
                                        )
                            
                    mean_merror = pd.Series(cv_results['binary_error-mean']).min()
                    boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()
        
                    if mean_merror < min_merror:
                        min_merror = mean_merror
                        best_params['max_bin']= max_bin
                        best_params['min_data_in_leaf'] = min_data_in_leaf
        
        cv_params['min_data_in_leaf'] = best_params['min_data_in_leaf']
        cv_params['max_bin'] = best_params['max_bin']
    elif cv_type=='bagging_fraction':
        logging.debug("调参3：降低过拟合")
        for feature_fraction in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
            for bagging_fraction in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
                for bagging_freq in range(0,50,5):
                    cv_params['feature_fraction'] = feature_fraction
                    cv_params['bagging_fraction'] = bagging_fraction
                    cv_params['bagging_freq'] = bagging_freq
                    
                    cv_results = lgb.cv(
                                        cv_params,
                                        lgb_train,
                                        seed=42,
                                        nfold=3,
                                        metrics=['binary_error'],
                                        early_stopping_rounds=3,
                                        verbose_eval=True
                                        )
                            
                    mean_merror = pd.Series(cv_results['binary_error-mean']).min()
                    boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()
        
                    if mean_merror < min_merror:
                        min_merror = mean_merror
                        best_params['feature_fraction'] = feature_fraction
                        best_params['bagging_fraction'] = bagging_fraction
                        best_params['bagging_freq'] = bagging_freq
        
        cv_params['feature_fraction'] = best_params['feature_fraction']
        cv_params['bagging_fraction'] = best_params['bagging_fraction']
        cv_params['bagging_freq'] = best_params['bagging_freq']
    elif cv_type=='lambda':
        logging.debug("调参4：降低过拟合")
        for lambda_l1 in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
            for lambda_l2 in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
                for min_split_gain in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
                    cv_params['lambda_l1'] = lambda_l1
                    cv_params['lambda_l2'] = lambda_l2
                    cv_params['min_split_gain'] = min_split_gain
                    
                    cv_results = lgb.cv(
                                        cv_params,
                                        lgb_train,
                                        seed=42,
                                        nfold=3,
                                        metrics=['binary_error'],
                                        early_stopping_rounds=3,
                                        verbose_eval=True
                                        )
                            
                    mean_merror = pd.Series(cv_results['binary_error-mean']).min()
                    boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()
        
                    if mean_merror < min_merror:
                        min_merror = mean_merror
                        best_params['lambda_l1'] = lambda_l1
                        best_params['lambda_l2'] = lambda_l2
                        best_params['min_split_gain'] = min_split_gain
        
        cv_params['lambda_l1'] = best_params['lambda_l1']
        cv_params['lambda_l2'] = best_params['lambda_l2']
        cv_params['min_split_gain'] = best_params['min_split_gain']
        
    logging.debug(cv_params)

def done(istrain=True):
    train_save,val_save,test_save,val_x,val_y = lightgbm_data_get(FLAGS.src_test_path)
    op=['max_depth','max_bin','bagging_fraction','lambda']
    ### 开始训练
    logging.debug('设置参数')
    if istrain:
        for oper in op:
            logging.debug("CV:"+oper)
            modelfit_cv(train_save,cv_type=oper)
            ret=dump(cv_params, FLAGS.tmp_data_path+'cv_params_'+oper+'lgbm.joblib_dat') 
        logging.debug("开始训练")
        gbm = lgb.train(cv_params,                     # 参数字典
                        train_save,                  # 训练集
                        num_boost_round=2000,       # 迭代次数
                        valid_sets=val_save,        # 验证集
                        early_stopping_rounds=30) # 早停系数

        
        logging.debug("to save validation predictions ...")
        ret=dump(gbm, FLAGS.tmp_data_path+'1-lgbm.model.joblib_dat') 
        logging.debug(ret)
        
        
        ### 验证
        logging.debug ("验证")
        preds_offline = gbm.predict(val_x, num_iteration=gbm.best_iteration) # 输出概率

        logging.debug('log_loss', log_loss(val_y, preds_offline))
        
    else:
        gbm = load(FLAGS.tmp_data_path+'1-lgbm.model.joblib_dat')
        
        ### 线下预测
        logging.debug ("预测")
        dtrain_predprob = gbm.predict(test_save, num_iteration=gbm.best_iteration) # 输出概率
        
        logging.debug(dtrain_predprob)
        y_pred = [round(value,4) for value in dtrain_predprob]
        logging.debug('-'*30)
        y_pred=np.array(y_pred).reshape(-1,1)
        logging.debug(y_pred.shape)
        test_id=pd.read_csv(FLAGS.tmp_data_path+'test_id.csv')
        logging.debug(test_id['id'].shape)
        test_id['id']=test_id['id'].map(int)
        test_id['click']=y_pred
        test_id.to_csv(FLAGS.tmp_data_path+'1-lgbm.test.csv',index=False)
        
        ### 特征选择
        df = pd.DataFrame(val_x.columns.tolist(), columns=['feature'])
        df['importance']=list(gbm.feature_importance())                           # 特征分数
        df = df.sort_values(by='importance',ascending=False)                      # 特征排序
        df.to_csv(FLAGS.tmp_data_path+'feature_score.csv',index=None,encoding='utf-8') # 保存分数
        
        
if __name__ == "__main__":
    done()
    done(False)
        

