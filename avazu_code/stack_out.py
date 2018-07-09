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


blending_w = {'rf': .075, 'xgb': .425, 'lgbm': .625, 'fm': .525}

total_w = 0
pred = 0


pred += xgb_pred['click'] * blending_w['xgb']
total_w += blending_w['xgb']
pred += lgbm_pred['click'] * blending_w['lgbm']
total_w += blending_w['lgbm']


pred /= total_w

y_pred = [round(value,4) for value in pred]
y_pred=np.array(y_pred).reshape(-1,1)
test_id=pd.read_csv(FLAGS.tmp_data_path+'test_id.csv')
logging.debug(test_id['id'].shape)
test_id['id']=test_id['id'].map(int)
test_id['click']=y_pred
test_id.to_csv(FLAGS.tmp_data_path+'1-stack.test.csv',index=False)
logging.debug( "=" * 80)
logging.debug( "Training complted and submission file 1-stack.test.csv created.")
logging.debug( "=" * 80)
