import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
from sklearn.utils import check_random_state 
import pylab 
import sys
import time
import utils
from utils import *
import os

from joblib import dump, load
sys.path.append('..')

from ml.ml_utils import *
from data_preprocessing import *

import logging


from flags import FLAGS, unparsed


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)

#rf_pred = load(utils.tmp_data_path + '1-fin-xgboost.test.csv')
xgb_pred = pd.read_csv(FLAGS.tmp_data_path + '1-fin-xgboost.test.csv')
lgbm_pred = pd.read_csv(FLAGS.tmp_data_path + '1-799-lgbm.test.csv')
logging.debug( "src data loaded")
pred=[]
logging.debug( xgb_pred.shape)

def select_pred(xgb_pred_value,lgbm_pred_value):
    if xgb_pred_value>=0.5 and lgbm_pred_value >=0.5:
        pred_value=max(xgb_pred_value,lgbm_pred_value)
    elif xgb_pred_value <0.5 and lgbm_pred_value <0.5:
        pred_value=min(xgb_pred_value,lgbm_pred_value)
    else:
        if abs(xgb_pred_value-0.5)>=abs(lgbm_pred_value-0.5):
            pred_value=xgb_pred_value
        else:
            pred_value=lgbm_pred_value
    if pred_value < 0.2:
        pred_value=0.0
    if pred_value > 0.9:
        pred_value=1.0
    return pred_value

new_pd=pd.merge(xgb_pred,lgbm_pred,on='id')
logging.debug( new_pd.head(5))
new_pd['pred']=new_pd.apply(lambda row: select_pred(row['click_x'], row['click_y']), axis=1)
pred=np.array(pred).reshape(-1,1)
test_id=pd.read_csv(FLAGS.tmp_data_path+'test_id.csv')
logging.debug(test_id['id'].shape)
test_id['id']=test_id['id'].map(int)
test_id['click']=new_pd['pred']
test_id.to_csv(FLAGS.tmp_data_path+'1-select.test.csv',index=False)
logging.debug( "=" * 80)
logging.debug( "Training complted and submission file 1-select.test.csv created.")
logging.debug( "=" * 80)
