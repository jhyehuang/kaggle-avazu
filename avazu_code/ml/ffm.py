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

import xlearn as xl

sys.path.append(utils.xgb_path)

import xgboost as xgb
import logging

import logging


from flags import FLAGS, unparsed


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)

param = {'task':'binary', 'lr':0.2, 'lambda':0.002,'epoch':10,
         'metric':['acc','log_loss'],
         'opt':'ftrl'}

param = {'task':'binary', 'lr':0.2}
param = {'task':'binary', 'lr':0.5}
param = {'task':'binary', 'lr':0.01}
param = {'task':'binary', 'lr':0.2, 'lambda':0.01}
param = {'task':'binary', 'lr':0.2, 'lambda':0.02}
param = {'task':'binary', 'lr':0.2, 'lambda':0.002}
param = {'alpha':0.002, 'beta':0.8, 'lambda_1':0.001, 'lambda_2': 1.0}
param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'k':2}
param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'k':4}
param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'k':5}
param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'k':8}
def done(istrain=True):
    ### 开始训练
    logging.debug('设置参数')
    if istrain:
        for i in [100,799,1537]:
            X_train_part, X_val, y_train_part, y_val = ffm_data_get_train(100)
            logging.debug("开始训练")
            try:
                init_model=load(FLAGS.out_data_path+'1-'+str(i)+'-ffm_model.model.joblib_dat')
            except:
                init_model=None
                
            ffm_model = xl.FFMModel(param)
            ffm_model.fit(X_train_part,y_train_part, eval_set=[X_val, y_val])
            
            logging.debug("to save validation predictions ...")
            ret=dump(gbm, FLAGS.out_data_path+'1-'+str(i)+'-ffm_model.model.joblib_dat') 
            logging.debug(ret)
            
        
        
            ### 验证
            logging.debug ("验证")
            preds_offline = ffm_model.predict(val_x) # 输出概率
    
            logging.debug('log_loss:')
            logging.debug(log_loss(val_y, preds_offline))
                        
            del X_train_part, X_val, y_train_part, y_val
        
    else:
        for i in [100,799,1537]:
            ffm_model = load(FLAGS.out_data_path+'1-'+str(i)+'-ffm_model.model.joblib_dat')
    #        logging.debug(gbm.get_params())
            ### 线下预测
            test_save=ffm_data_get_test()
            logging.debug ("预测")
            dtrain_predprob = ffm_model.predict(test_save) # 输出概率
            
            logging.debug(dtrain_predprob)
            y_pred = [round(value,4) for value in dtrain_predprob]
            logging.debug('-'*30)
            y_pred=np.array(y_pred).reshape(-1,1)
            logging.debug(y_pred.shape)
            test_id=pd.read_csv(FLAGS.test_id_path+'test_id.csv')
            logging.debug(test_id['id'].shape)
            test_id['id']=test_id['id'].map(int)
            test_id['click']=y_pred
            test_id.to_csv(FLAGS.out_data_path+'1-'+str(i)+'-ffm_model.test.csv',index=False)
            
    
            del test_save
        
if __name__ == "__main__":
    done()
    done(False)

