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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


import logging


from flags import FLAGS, unparsed


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)
#logging.debug(FLAGS)
train_save,test_save = lr_data_get(FLAGS.src_test_path)
logging.debug(train_save.shape)
y_train = train_save['click']
train_save.drop('click',axis=1,inplace=True)
X_train = train_save

test_save.drop('click',axis=1,inplace=True)
X_test=test_save

y_train[:500]=1
#X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=4)

n_trees = FLAGS.xgb_n_trees


param = FLAGS.gbdt_param

np.random.seed(999)
train_idx = np.random.random_integers(0, 3, X_train.shape[0])
test_idx = np.random.random_integers(0, 3, X_train.shape[0])
logging.debug(train_idx.shape)
predv_xgb = []
ctr = 0
model_LR= LogisticRegression()
for idx in [0, 1, 2, 3]:
    filter1 = np.logical_and(train_idx== idx , True)
    logging.debug(filter1)
    filter_v1 = np.logical_and(test_idx== idx , True)
    xt1 = X_train.loc[filter1, :]
    yt1 = y_train.loc[filter1]
    xt2=X_train.loc[filter_v1,:]
    yt2=y_train.loc[filter_v1]
    logging.debug(yt1.unique())
    if xt1.shape[0] <=0 or xt1.shape[0] != yt1.shape[0]:
        logging.debug(xt1.shape, yt1.shape)
        raise ValueError('wrong shape!')
    logging.debug(xt1.shape)
    logging.debug(yt1.shape)
    model_LR.fit(xt1,yt1)
    y_prob = model_LR.predict_proba(xt2)[:,1] # This will give you positive class prediction probabilities  
    xgb_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
       
    #accuracy 
    logging.debug(yt2.shape)
    logging.debug(y_prob.shape)
    print ('The accuary of default Logistic Regression is',model_LR.score(xt2, xgb_pred)) 
#    predv_xgb+=(xgb_pred)
    #xgb_list[rseed] = xgb1
    print ('The AUC of default Logistic Regression is', roc_auc_score(yt2,y_prob))

    logging.debug(ctr)

    ctr += 1
#    predv_xgb = xgb_pred+predv_xgb
#    logging.debug(predv_xgb.shape)
    logging.debug('-'*30)
    logging.debug( str(ctr))
    logging.debug(str(logloss(y_prob , yt2)))

#logging.debug("to save validation predictions ...")
#dump(predv_xgb , FLAGS.tmp_data_path + 'xgb_pred_v.joblib_dat')

