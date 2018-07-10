import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
import pylab 
import sys
import time
import os
sys.path.append('..')
import utils
from ml_utils import *
from data_preprocessing import *
from joblib import dump, load, Parallel, delayed

import xlearn as xl


import logging



from flags import FLAGS, unparsed


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)

param = {'task':'binary', 'lr':0.2, 'lambda':0.002,'epoch':10,
         'metric':['acc','log_loss'],'k':4,
         'opt':'ftrl'}

#param = {'task':'binary', 'lr':0.2}
#param = {'task':'binary', 'lr':0.5}
#param = {'task':'binary', 'lr':0.01}
#param = {'task':'binary', 'lr':0.2, 'lambda':0.01}
#param = {'task':'binary', 'lr':0.2, 'lambda':0.02}
#param = {'task':'binary', 'lr':0.2, 'lambda':0.002}
ftrl_param = {'alpha':0.002, 'beta':0.8, 'lambda_1':0.001, 'lambda_2': 1.0}
#param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'k':2}
#param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'k':4}
#param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'k':5}
#param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'k':8}

param.update(ftrl_param)

def done(istrain=True):
    ### 开始训练
    logging.debug('设置参数')
    if istrain:
        logging.debug("开始训练")                
        ffm_model = xl.FFMModel(model_type='ffm')
        ffm_model.fit(FLAGS.tmp_data_path +'ont_hot_train.libffm.csv', eval_set=FLAGS.tmp_data_path +'ont_hot_train.libffm.csv')
        
        logging.debug("to save validation predictions ...")
        ret=dump(ffm_model, FLAGS.out_data_path+'1-'+'-ffm_model.model.joblib_dat') 
        logging.debug(ret)
        logging.debug(ffm_model.weights)        
    else:

        ffm_model = load(FLAGS.out_data_path+'1-'+'-ffm_model.model.joblib_dat')
#        logging.debug(gbm.get_params())
        ### 线下预测
        test_save=FLAGS.tmp_data_path +'ont_hot_test.libffm.csv'
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
        test_id.to_csv(FLAGS.out_data_path+'1-'+'-ffm_model.test.csv',index=False)
        
        
if __name__ == "__main__":
    done()
#    done(False)

