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

sys.path.append(utils.xgb_path)

import xgboost as xgb
import logging

from flags import parse_args
FLAGS, unparsed = parse_args()

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)


train_save = load(FLAGS.tmp_data_path +FLAGS.train_job_name +'.joblib_dat')
x = train_save[FLAGS.train_col]
y = train_save[FLAGS.train_target]



rseed = 0
xgb_eta = .3
tvh = utils.tvh
n_passes = 5
n_trees = 40
n_iter = 7
n_threads = 8
nr_factor = 4

i = 1
while i < len(sys.argv):
    if sys.argv[i] == '-rseed':
        i += 1
        rseed = int(sys.argv[i])
    elif sys.argv[i] == '-passes':
        i += 1
        n_passes = int(sys.argv[i])
    else:
        raise ValueError("unrecognized parameter [" + sys.argv[i] + "]")
    
    i += 1

learning_rate = .1

path1 = utils.tmp_data_path
param_names = '_r' + str(rseed)
 
fn_t = path1 + 'fm_' + param_names + '_t.txt'
fn_v = path1 + 'fm_' + param_names + '_v.txt'

test_day = 30
if tvh == 'Y':
    test_day = 31

def build_data():

    np.random.seed(rseed)
    nn = x.shape[0]
    r1 = np.random.uniform(0, 1, nn)


    filter1 = np.logical_and(x, np.logical_and(r1 < 0.25, True))
    filter_v1 = np.logical_and(x, np.logical_and(r1 < 0.25, True))

    xt1 = x[filter1, :]
    yt1 = y[filter1]
    if xt1.shape[0] <=0 or xt1.shape[0] != yt1.shape[0]:
        logging.debug(xt1.shape, yt1.shape)
        raise ValueError('wrong shape!')
    dtrain = xgb.DMatrix(xt1, label=yt1)
    dvalid = xgb.DMatrix(x[filter_v1], label=y[filter_v1])
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    logging.debug(xt1.shape, yt1.shape)

    param = {'max_depth':6, 'eta':.5, 'objective':'binary:logistic', 'verbose':0,
             'subsample':1.0, 'min_child_weight':50, 'gamma':0,
             'nthread': 16, 'colsample_bytree':.5, 'base_score':0.16, 'seed': rseed}

    plst = list(param.items()) + [('eval_metric', 'logloss')]
    xgb_test_basis_d6 = xgb.train(plst, dtrain, n_trees, watchlist)


    dtv = xgb.DMatrix(x)
    xgb_leaves = xgb_test_basis_d6.predict(dtv, pred_leaf = True)

    t0 = pd.DataFrame({'click': y})
    logging.debug(xgb_leaves.shape)
    for i in range(n_trees):
        pred2 = xgb_leaves[:, i]
        logging.debug(i, np.unique(pred2).size)
        t0['xgb_basis'+str(i)] = pred2

    t3a_save = load(utils.tmp_data_path + 't3a.joblib_dat')
    t3a = t3a_save['t3a']

    idx_base = 0
    for vn in ['xgb_basis' + str(i) for i in range(n_trees)]:
        _cat = np.asarray(t0[vn].astype('category').values.codes, dtype='int32')
        _cat1 = _cat + idx_base
        logging.debug(vn, idx_base, _cat1.min(), _cat1.max(), np.unique(_cat).size)
        t3a[vn] = _cat1
        idx_base += _cat.max() + 1



    t3a.ix[filter1,:].to_csv(open(fn_t, 'w'), sep='\t', header=False, index=False)
    t3a.ix[filter_v1,:].to_csv(open(fn_v, 'w'), sep='\t', header=False, index=False)


build_data()
import gc
gc.collect()


fm_cmd = utils.fm_path + ' -k ' + str(nr_factor) + ' -t ' + str(n_iter) + ' -s '+ str(n_threads) + ' '
fm_cmd += ' -d ' + str(rseed) + ' -r ' + str(learning_rate) + ' ' + fn_v + ' ' + fn_t

logging.debug(fm_cmd)
os.system(fm_cmd)

os.system("rm " + fn_t)
os.system("rm " + fn_v)

