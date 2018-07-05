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
logging.debug('设置参数')
cv_params = {
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric': 'binary_logloss',
#          'num_trees':100,
            'device': 'gpu',
            'gpu_platform_id': -1,
            'gpu_device_id': -1
}


gpu_params = {
            'boosting_type': 'gbdt',
#            'boosting': 'dart',
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            
            'learning_rate': 0.01,
            'num_leaves':55,
            'max_depth':7,
            
            'max_bin':4,
            'min_data_in_leaf':55,
            
            'feature_fraction': 0.4,
            'bagging_fraction': 0.4,
            'bagging_freq':2,

            'lambda_l1': 0.7,
            'lambda_l2': 0.7,
            'min_split_gain': 0.01,
            'sparse_threshold': 1.0,
#            'device': 'gpu',
#            'gpu_platform_id': -1,
#            'gpu_device_id': -1,
#            'gpu_use_dp': False
}


#cv_params.update(gpu_params)
### 交叉验证(调参)


def modelfit_cv(lgb_train,cv_type='max_depth',):
    min_merror = float('Inf')
    logging.debug('交叉验证')
    best_params = {}
    early_stopping_dict={'mean_merror':min_merror,'times':0}
    if cv_type=='num_trees':
        # 准确率
        logging.debug("num_trees")
        for num_trees in range(100,300,5):
                cv_params['num_trees'] = num_trees
                cv_results = lgb.cv(
                                    cv_params,
                                    lgb_train,
                                    seed=2018,
                                    nfold=3,
                                    metrics=['auc', 'binary_logloss'],
                                    early_stopping_rounds=3,
                                    verbose_eval=True
                                    )
                    
                mean_merror = pd.Series(cv_results['binary_logloss-mean']).min()
                boost_rounds = pd.Series(cv_results['binary_logloss-mean']).argmin()
                logging.debug("now num_trees="+str(num_trees))
                logging.debug("boost_rounds="+str(boost_rounds))
                logging.debug("mean_merror="+str(mean_merror))
                if mean_merror < early_stopping_dict['mean_merror']:
                    early_stopping_dict['mean_merror']=mean_merror
                    early_stopping_dict['times']=0
                    if mean_merror < min_merror:
                        min_merror = mean_merror
                        best_params['num_trees'] = num_trees
                else:
                    early_stopping_dict['times']+=1
                    
#                if early_stopping_dict['times']>3:
#                    early_stopping_dict['times']=0
#                    break
#            if early_stopping_dict['times']>3:
#                break

                    
        cv_params['num_trees'] = best_params['num_trees']
        logging.debug("best_params['num_trees']="+str(best_params['num_trees']))
    
    
    elif cv_type=='max_depth':
        # 准确率
        logging.debug("调参1：提高准确率")
        for num_leaves in range(30,60,5):
            for max_depth in range(4,8,1):
                cv_params['num_leaves'] = num_leaves
                cv_params['max_depth'] = max_depth
        
                cv_results = lgb.cv(
                                    cv_params,
                                    lgb_train,
                                    seed=2018,
                                    nfold=3,
                                    metrics=['auc', 'binary_logloss'],
                                    early_stopping_rounds=10,
                                    verbose_eval=True
                                    )
                    
                mean_merror = pd.Series(cv_results['binary_logloss-mean']).min()
                boost_rounds = pd.Series(cv_results['binary_logloss-mean']).argmin()
                logging.debug("boost_rounds="+str(boost_rounds))
                logging.debug("mean_merror="+str(mean_merror))
                if mean_merror < early_stopping_dict['mean_merror']:
                    early_stopping_dict['mean_merror']=mean_merror
                    early_stopping_dict['times']=0
                    if mean_merror < min_merror:
                        min_merror = mean_merror
                        best_params['num_leaves'] = num_leaves
                        best_params['max_depth'] = max_depth
                else:
                    early_stopping_dict['times']+=1
                    
#                if early_stopping_dict['times']>3:
#                    early_stopping_dict['times']=0
#                    break
#            if early_stopping_dict['times']>3:
#                break

                    
        cv_params['num_leaves'] = best_params['num_leaves']
        cv_params['max_depth'] = best_params['max_depth']
    elif cv_type=='max_bin':
        # 过拟合
        logging.debug("调参2：降低过拟合")
        for max_bin in range(4,80,5):
            for min_data_in_leaf in range(10,60,5):
                cv_params['max_bin'] = max_bin
                cv_params['min_data_in_leaf'] = min_data_in_leaf
                
                cv_results = lgb.cv(
                                    cv_params,
                                    lgb_train,
                                    seed=42,
                                    nfold=3,
                                    metrics=['auc', 'binary_logloss'],
                                    early_stopping_rounds=10,
                                    verbose_eval=True
                                    )
                        
                mean_merror = pd.Series(cv_results['binary_logloss-mean']).min()
                boost_rounds = pd.Series(cv_results['binary_logloss-mean']).argmin()
                logging.debug("boost_rounds="+str(boost_rounds))
                logging.debug("mean_merror="+str(mean_merror))
                logging.debug("best_params['max_bin']="+str(max_bin))
                logging.debug("best_params['min_data_in_leaf']="+str(min_data_in_leaf))
        
                if mean_merror < early_stopping_dict['mean_merror']:
                    early_stopping_dict['mean_merror']=mean_merror
                    early_stopping_dict['times']=0

                    best_params['max_bin']= max_bin
                    best_params['min_data_in_leaf'] = min_data_in_leaf
                else:
                    early_stopping_dict['times']+=1
                    
#                if early_stopping_dict['times']>3:
#                    early_stopping_dict['times']=0
#                    break
#            if early_stopping_dict['times']>3:
#                break
        
        
        cv_params['min_data_in_leaf'] = best_params['min_data_in_leaf']
        cv_params['max_bin'] = best_params['max_bin']
    elif cv_type=='bagging_fraction':
        logging.debug("调参3：降低过拟合")
        for feature_fraction in [0.1,0.2,0.3,0.4]:
            for bagging_fraction in [0.1,0.2,0.3,0.4,0.5]:
                for bagging_freq in range(2,7,1):
                    cv_params['feature_fraction'] = feature_fraction
                    cv_params['bagging_fraction'] = bagging_fraction
                    cv_params['bagging_freq'] = bagging_freq
                    
                    cv_results = lgb.cv(
                                        cv_params,
                                        lgb_train,
                                        seed=42,
                                        nfold=3,
                                        metrics=['auc', 'binary_logloss'],
                                        early_stopping_rounds=10,
                                        verbose_eval=True
                                        )
                            
                    mean_merror = pd.Series(cv_results['binary_logloss-mean']).min()
                    boost_rounds = pd.Series(cv_results['binary_logloss-mean']).argmin()
                    logging.debug("boost_rounds="+str(boost_rounds))
                    logging.debug("mean_merror="+str(mean_merror))

                    logging.debug("best_params['feature_fraction']="+str(feature_fraction))
                    logging.debug("best_params['bagging_fraction']="+str(bagging_fraction))
                    logging.debug("best_params['bagging_freq']="+str(bagging_freq))
                
                    if mean_merror < early_stopping_dict['mean_merror']:
                        early_stopping_dict['mean_merror']=mean_merror
                        early_stopping_dict['times']=0
    
                        best_params['feature_fraction'] = feature_fraction
                        best_params['bagging_fraction'] = bagging_fraction
                        best_params['bagging_freq'] = bagging_freq
                    else:
                        early_stopping_dict['times']+=1
                    
#                    if early_stopping_dict['times']>3:
#                        early_stopping_dict['times']=0
#                        break
#                if early_stopping_dict['times']>3:
#                    break
#            if early_stopping_dict['times']>3:
#                break
        
        
        cv_params['feature_fraction'] = best_params['feature_fraction']
        cv_params['bagging_fraction'] = best_params['bagging_fraction']
        cv_params['bagging_freq'] = best_params['bagging_freq']
    elif cv_type=='lambda':
        logging.debug("调参4：降低过拟合")
        for lambda_l1 in [0.7,0.8,0.9,1.0]:
            for lambda_l2 in [0.4,0.5,0.6,0.7]:
                for min_split_gain in [0.01,0.05,0.07,0.1,0.2]:
                    cv_params['lambda_l1'] = lambda_l1
                    cv_params['lambda_l2'] = lambda_l2
                    cv_params['min_split_gain'] = min_split_gain
                    
                    cv_results = lgb.cv(
                                        cv_params,
                                        lgb_train,
                                        seed=42,
                                        nfold=3,
                                        metrics=['auc', 'binary_logloss'],
                                        early_stopping_rounds=10,
                                        verbose_eval=True
                                        )
                            
                    mean_merror = pd.Series(cv_results['binary_logloss-mean']).min()
                    boost_rounds = pd.Series(cv_results['binary_logloss-mean']).argmin()
                    logging.debug("boost_rounds="+str(boost_rounds))
                    logging.debug("mean_merror="+str(mean_merror))

                    logging.debug("best_params['lambda_l1']="+str(lambda_l1))
                    logging.debug("best_params['lambda_l2']="+str(lambda_l2))
                    logging.debug("best_params['min_split_gain']="+str(min_split_gain))
        
                    if mean_merror < early_stopping_dict['mean_merror']:
                        early_stopping_dict['mean_merror']=mean_merror
                        early_stopping_dict['times']=0
    
                        best_params['lambda_l1'] = lambda_l1
                        best_params['lambda_l2'] = lambda_l2
                        best_params['min_split_gain'] = min_split_gain
                    else:
                        early_stopping_dict['times']+=1
                    
#                    if early_stopping_dict['times']>3:
#                        early_stopping_dict['times']=0
#                        break
#                if early_stopping_dict['times']>3:
#                    break
#            if early_stopping_dict['times']>3:
#                break
        cv_params['lambda_l1'] = best_params['lambda_l1']
        cv_params['lambda_l2'] = best_params['lambda_l2']
        cv_params['min_split_gain'] = best_params['min_split_gain']
        
    logging.debug(cv_params)

def done(istrain=True):
    
#    op=['num_trees','max_depth','max_bin','bagging_fraction','lambda']
#    cv_params['num_leaves'] = 50
#    cv_params['max_depth'] = 6
#    op=['max_bin','bagging_fraction','lambda']
    op=['num_trees']
    ### 开始训练
    logging.debug('设置参数')
    if istrain:
        train_save,val_save,val_x,val_y = tiny_lightgbm_data_get_train()
        for oper in op:
            logging.debug("CV:"+oper)
            modelfit_cv(train_save,cv_type=oper)
            ret=dump(cv_params, FLAGS.out_data_path+'cv_params_'+oper+'lgbm.joblib_dat') 
        logging.debug("开始训练")
        gbm = lgb.train(cv_params,                     # 参数字典
                        train_save,                  # 训练集
                        num_boost_round=100,       # 迭代次数
                        valid_sets=val_save,        # 验证集
                        early_stopping_rounds=30) # 早停系数

        
        logging.debug("to save validation predictions ...")
        ret=dump(gbm, FLAGS.out_data_path+'1-lgbm.model.joblib_dat') 
        logging.debug(ret)
        
        
        ### 验证
        logging.debug ("验证")
        preds_offline = gbm.predict(val_x, num_iteration=gbm.best_iteration) # 输出概率

        logging.debug('log_loss:')
        logging.debug(log_loss(val_y, preds_offline))
        
        ### 特征选择
        df = pd.DataFrame(val_x.columns.tolist(), columns=['feature'])
        df['importance']=list(gbm.feature_importance())                           # 特征分数
        df = df.sort_values(by='importance',ascending=False)                      # 特征排序
        df.to_csv(FLAGS.out_data_path+'feature_score.csv',index=None,encoding='utf-8') # 保存分数
        
        del train_save,val_save,val_x,val_y
        
    else:
        gbm = load(FLAGS.out_data_path+'1-lgbm.model.joblib_dat')
        
        ### 线下预测
        test_save=tiny_lightgbm_data_get_test()
        logging.debug ("预测")
        dtrain_predprob = gbm.predict(test_save, num_iteration=gbm.best_iteration) # 输出概率
        
        logging.debug(dtrain_predprob)
        y_pred = [round(value,4) for value in dtrain_predprob]
        logging.debug('-'*30)
        y_pred=np.array(y_pred).reshape(-1,1)
        logging.debug(y_pred.shape)
        test_id=pd.read_csv(FLAGS.test_id_path+'test_id.csv')
        logging.debug(test_id['id'].shape)
        test_id['id']=test_id['id'].map(int)
        test_id['click']=y_pred
        test_id.to_csv(FLAGS.out_data_path+'1-lgbm.test.csv',index=False)
        

        del test_save
        
if __name__ == "__main__":
#    done()
    done(False)
        

