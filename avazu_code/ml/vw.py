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
'''
vw  ---- vowpal_wabbit
'''



logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)

train_save = load(FLAGS.tmp_data_path +FLAGS.train_job_name +'.joblib_dat')
x = train_save[FLAGS.train_col]
y = train_save[FLAGS.train_target]


rseed = 0
xgb_eta = .3
tvh = utils.tvh
n_passes = 4

i = 1
while i < len(sys.argv):
    if sys.argv[i] == '-rseed':
        i += 1
        rseed = int(sys.argv[i])
    else:
        raise ValueError("unrecognized parameter [" + sys.argv[i] + "]")
    
    i += 1


file_name1 = '_r' + str(rseed)

path1 = utils.tmp_data_path
fn_t = path1 + 'vwV12_' + file_name1 + '_train.txt'
fn_v = path1 + 'vwV12_' + file_name1 + '_test.txt'


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

    n_trees = 30
    n_parallel_tree = 1

    param = {'max_depth':6, 'eta':xgb_eta, 'objective':'binary:logistic', 'verbose':1,
	     'subsample':1.0, 'min_child_weight':50, 'gamma':0,
	     'nthread': 16, 'colsample_bytree':.5, 'base_score':0.16, 'seed': rseed,
	     'num_parallel_tree': n_parallel_tree}

    plst = list(param.items()) + [('eval_metric', 'logloss')]
    xgb_test_basis_d6 = xgb.train(plst, dtrain, n_trees, watchlist)

    logging.debug("to score gbdt ...")

    dtv = xgb.DMatrix(x)
    xgb_leaves = xgb_test_basis_d6.predict(dtv, pred_leaf = True)
        
    t0 = pd.DataFrame({FLAGS.train_target: y})
    logging.debug(xgb_leaves.shape)
    for i in range(n_trees * n_parallel_tree):
        pred2 = xgb_leaves[:, i]
        logging.debug(i, np.unique(pred2).size)
        t0['xgb_basis'+str(i)] = pred2

    t3a_save = load(utils.tmp_data_path + 't3a.joblib_dat')

    t3a = t3a_save['t3a']
    idx_base = 0
    for vn in ['xgb_basis' + str(i) for i in range(30 * n_parallel_tree)]:
        _cat = np.asarray(t0[vn].astype('category').values.codes, dtype='int32')
        _cat1 = _cat + idx_base
        logging.debug(vn, idx_base, _cat1.min(), _cat1.max(), np.unique(_cat).size)
        t3a[vn] = _cat1
        idx_base += _cat.max() + 1


    t3a['idx'] = np.arange(t3a.shape[0])
    t3a.set_index('idx', inplace=True)

    logging.debug("to write training file, this may take a long time")
    import gzip
    t3a.ix[filter1,:].to_csv(open(fn_t, 'w'), sep=' ', header=False, index=False)

    os.system("gzip -f "+fn_t)

    logging.debug("to write test file, this shouldn't take too long")
    t3a.ix[filter_v1, :].to_csv(open(fn_v, 'w'), sep=' ', header=False, index=False)
    os.system("gzip -f "+fn_v)


build_data()


holdout_str = " --holdout_period 7 "
    
mdl_name = 'vw' + file_name1 + ".mdl"
vw_cmd_str = utils.vw_path + fn_t + ".gz --random_seed " + str(rseed) + " " + \
"--passes " + str(n_passes) + " -c --progress 1000000 --loss_function logistic -b 25 " +  holdout_str + \
"--l2 1e-7 -q CS -q CM -q MS -l .1 --power_t .5 -q NM -q NS --decay_learning_rate .75 --hash all " + \
" -q SX -q MX -q SY -q MY -q SZ -q MZ -q NV -q MV -q VX -q VY -q VZ" + \
" --ignore H -f " + mdl_name + " -k --compressed"
logging.debug(vw_cmd_str)
os.system(vw_cmd_str)

vw_cmd_str = utils.vw_path + fn_v + ".gz --hash all " + \
    "-i " + mdl_name + " -p " + fn_v + "_pred.txt -t --loss_function logistic --progress 200000"
logging.debug(vw_cmd_str)
os.system(vw_cmd_str)

