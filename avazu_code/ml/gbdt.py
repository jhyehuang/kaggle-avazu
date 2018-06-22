import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
from sklearn.utils import check_random_state 
from sklearn.model_selection import train_test_split
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
def done(istrain=True):
    train_save,test_save = gdbt_data_get(FLAGS.src_test_path)
    print(train_save.shape)
    y_train = train_save['click']
    train_save.drop('click',axis=1,inplace=True)
    X_train = train_save
    
    test_save.drop('click',axis=1,inplace=True)
    X_test=test_save
    if istrain:
        dtrain = xgb.DMatrix(X_train, label=y_train)

        plst = list(param.items()) + [('eval_metric', 'logloss')]
        xgb1 = xgb.cv(plst, dtrain, folds =5,metrics='logloss')
        
        logging.debug("to save validation predictions ...")
        xgb.save_model('xgb.model')
        pd.DataFrame(xgb1).to_csv(FLAGS.tmp_data_path+'1-gdbt.csv')
    else:
        xgb1 = xgb.Booster(model_file='xgb.model')
#        xgb1=pd.read_csv(FLAGS.tmp_data_path+'1-gdbt.csv')
        dtest = xgb.DMatrix(X_test)
        xgb_pred = xgb1.predict(dtest)
        y_pred = [round(value) for value in xgb_pred]
        logging.debug('-'*30)
        logging.debug(y_pred)
        
if __name__ == "__main__":
    done()
    done(False)
        

