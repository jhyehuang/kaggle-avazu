import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
from sklearn.utils import check_random_state 
import pylab 
import sys
import time
sys.path.append('/home/zzhang/Downloads/xgboost/wrapper')
import xgboost as xgb
from joblib import dump, load, Parallel, delayed
import utils
from utils import *


raw_data_path = utils.raw_data_path
tmp_data_path = utils.tmp_data_path


trainorg0 = pd.read_csv(open(raw_data_path + "train", "ra"))
h0org = pd.read_csv(open(raw_data_path + "test", "ra"))

#下采样：sample_pct=1/0.05
#原始样本约40M, 40M*0.05 = 2M
if utils.sample_pct < 1.0:
    np.random.seed(999)
    r1 = np.random.uniform(0, 1, trainorg0.shape[0])  #产生0～40M的随机数
    trainorg0 = trainorg0.ix[r1 < utils.sample_pct, :]
    print "testing with small sample of training data, ", trainorg0.shape

#2a
h0org['click'] = 0  #测试样本加一列click，初始化为0
#将训练样本和测试样本连接，一起进行特征工程
trainorg = pd.concat([trainorg0, h0org])
print "finished loading raw data, ", trainorg.shape

print "to add some basic features ..."
#处理hour特征，格式为YYMMDDHH
#数据为21-31，共11天的数据
trainorg['day']=np.round(trainorg.hour % 10000 / 100)
trainorg['hour1'] = np.round(trainorg.hour % 100)
trainorg['day_hour'] = (trainorg.day.values - 21) * 24 + trainorg.hour1.values
trainorg['day_hour_prev'] = trainorg['day_hour'] - 1
trainorg['day_hour_next'] = trainorg['day_hour'] + 1

trainorg['app_or_web'] = 0
#如果app_id='ecad2386',app_or_web=1
trainorg.ix[trainorg.app_id.values=='ecad2386', 'app_or_web'] = 1

train = trainorg

#串联app_id和site_id
train['app_site_id'] = np.add(train.app_id.values, train.site_id.values)


#2b，后验均值编码
print "to encode categorical features using mean responses from earlier days -- univariate"
sys.stdout.flush()

calc_exptv(train,  ['app_or_web'])

exptv_vn_list = ['app_site_id', 'as_domain', 'C14','C17', 'C21', 'device_model', 'device_ip', 'device_id', 'dev_ip_aw', 
                'app_site_model', 'site_model','app_model', 'dev_id_ip', 'C14_aw', 'C17_aw', 'C21_aw']

calc_exptv(train, exptv_vn_list)

calc_exptv(train, ['app_site_id'], add_count=True)


print "to encode categorical features using mean responses from earlier days -- multivariate"
vns = ['app_or_web',  'device_ip', 'app_site_id', 'device_model', 'app_site_model', 'C1', 'C14', 'C17', 'C21',
                        'device_type', 'device_conn_type','app_site_model_aw', 'dev_ip_app_site']

# 训练&测试
dftv = train.ix[np.logical_and(train.day.values >= 21, train.day.values < 32), ['click', 'day', 'id'] + vns].copy()

#串联两个特征成新的特征
dftv['app_site_model'] = np.add(dftv.device_model.values, dftv.app_site_id.values)
dftv['app_site_model_aw'] = np.add(dftv.app_site_model.values, dftv.app_or_web.astype('string').values)
dftv['dev_ip_app_site'] = np.add(dftv.device_ip.values, dftv.app_site_id.values)

#初始化
for vn in vns:
    dftv[vn] = dftv[vn].astype('category')
    print vn

#后验均值编码中的先验强度
n_ks = {'app_or_web': 100, 'app_site_id': 100, 'device_ip': 10, 'C14': 50, 'app_site_model': 50, 'device_model': 100, 'device_id': 50,
        'C17': 100, 'C21': 100, 'C1': 100, 'device_type': 100, 'device_conn_type': 100, 'banner_pos': 100,
        'app_site_model_aw': 100, 'dev_ip_app_site': 10 , 'device_model': 500}

#初始化
exp2_dict = {}
for vn in vns:
    exp2_dict[vn] = np.zeros(dftv.shape[0])

days_npa = dftv.day.values
    
for day_v in xrange(22, 32):
    # day_v之前的天，所以从22开始，作为训练集
    df1 = dftv.ix[np.logical_and(dftv.day.values < day_v, dftv.day.values < 31), :].copy()

    #当前天的记录，作为校验集
    df2 = dftv.ix[dftv.day.values == day_v, :]
    print "Validation day:", day_v, ", train data shape:", df1.shape, ", validation data shape:", df2.shape

    #每个样本的y的先验都等于平均click率
    pred_prev = df1.click.values.mean() * np.ones(df1.shape[0])


    for vn in vns:
        if 'exp2_'+vn in df1.columns:  #已经有了，丢弃重新计算
            df1.drop('exp2_'+vn, inplace=True, axis=1)

    for i in xrange(3):
        for vn in vns:
            p1 = calcLeaveOneOut2(df1, vn, 'click', n_ks[vn], 0, 0.25, mean0=pred_prev)
            pred = pred_prev * p1
            print day_v, i, vn, "change = ", ((pred - pred_prev)**2).mean()
            pred_prev = pred    

        #y的先验
        pred1 = df1.click.values.mean()
        for vn in vns:
            print "="*20, "merge", day_v, vn
            diff1 = mergeLeaveOneOut2(df1, df2, vn)
            pred1 *= diff1
            exp2_dict[vn][days_npa == day_v] = diff1
        
        pred1 *= df1.click.values.mean() / pred1.mean()
        print "logloss = ", logloss(pred1, df2.click.values)
        #print my_lift(pred1, None, df2.click.values, None, 20, fig_size=(10, 5))
        #plt.show()

for vn in vns:
    train['exp2_'+vn] = exp2_dict[vn]

#2c
print "to count prev/current/next hour by ip ..."
cntDualKey(train, 'device_ip', None, 'day_hour', 'day_hour_prev', fill_na=0)
cntDualKey(train, 'device_ip', None, 'day_hour', 'day_hour', fill_na=0)
cntDualKey(train, 'device_ip', None, 'day_hour', 'day_hour_next', fill_na=0)

print "to create day diffs"
train['pday'] = train.day - 1
calcDualKey(train, 'device_ip', None, 'day', 'pday', 'click', 10, None, True, True)
train['cnt_diff_device_ip_day_pday'] = train.cnt_device_ip_day.values  - train.cnt_device_ip_pday.values
train['hour1_web'] = train.hour1.values
train.ix[train.app_or_web.values==0, 'hour1_web'] = -1
train['app_cnt_by_dev_ip'] = my_grp_cnt(train.device_ip.values.astype('string'), train.app_id.values.astype('string'))


train['hour1'] = np.round(train.hour.values % 100)
train['cnt_diff_device_ip_day_pday'] = train.cnt_device_ip_day.values  - train.cnt_device_ip_pday.values

train['rank_dev_ip'] = my_grp_idx(train.device_ip.values.astype('string'), train.id.values.astype('string'))
train['rank_day_dev_ip'] = my_grp_idx(np.add(train.device_ip.values, train.day.astype('string').values).astype('string'), train.id.values.astype('string'))
train['rank_app_dev_ip'] = my_grp_idx(np.add(train.device_ip.values, train.app_id.values).astype('string'), train.id.values.astype('string'))


train['cnt_dev_ip'] = get_agg(train.device_ip.values, train.id, np.size)
train['cnt_dev_id'] = get_agg(train.device_id.values, train.id, np.size)

train['dev_id_cnt2'] = np.minimum(train.cnt_dev_id.astype('int32').values, 300)
train['dev_ip_cnt2'] = np.minimum(train.cnt_dev_ip.astype('int32').values, 300)

train['dev_id2plus'] = train.device_id.values
train.ix[train.cnt_dev_id.values == 1, 'dev_id2plus'] = '___only1'
train['dev_ip2plus'] = train.device_ip.values
train.ix[train.cnt_dev_ip.values == 1, 'dev_ip2plus'] = '___only1'

train['diff_cnt_dev_ip_hour_phour_aw2_prev'] = (train.cnt_device_ip_day_hour.values - train.cnt_device_ip_day_hour_prev.values) * ((train.app_or_web * 2 - 1)) 
train['diff_cnt_dev_ip_hour_phour_aw2_next'] = (train.cnt_device_ip_day_hour.values - train.cnt_device_ip_day_hour_next.values) * ((train.app_or_web * 2 - 1)) 


print "to save train ..."

dump(train, tmp_data_path + 'train.joblib_dat')


print "to generate traintv_mx .. "
app_or_web = None
_start_day = 22
list_param = ['C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'banner_pos', 'device_type', 'device_conn_type']
feature_list_dict = {}

feature_list_name = 'tvexp3'
feature_list_dict[feature_list_name] = list_param + \
                            ['exptv_' + vn for vn in ['app_site_id', 'as_domain', 
                             'C14','C17', 'C21', 'device_model', 'device_ip', 'device_id', 'dev_ip_aw', 
                             'dev_id_ip', 'C14_aw', 'C17_aw', 'C21_aw']] + \
                            ['cnt_diff_device_ip_day_pday', 
                             'app_cnt_by_dev_ip', 'cnt_device_ip_day_hour', 'app_or_web',
                             'rank_dev_ip', 'rank_day_dev_ip', 'rank_app_dev_ip',
                             'diff_cnt_dev_ip_hour_phour_aw2_prev', 'diff_cnt_dev_ip_hour_phour_aw2_next',
                             'exp2_device_ip', 'exp2_app_site_id', 'exp2_device_model', 'exp2_app_site_model',
                             'exp2_app_site_model_aw', 'exp2_dev_ip_app_site',
                             'cnt_dev_ip', 'cnt_dev_id', 'hour1_web']

filter_tv = np.logical_and(train.day.values >= _start_day, train.day.values < 31)
filter_t1 = np.logical_and(train.day.values < 30, filter_tv)
filter_v1 = np.logical_and(~filter_t1, filter_tv)    
    
print filter_tv.sum()


for vn in feature_list_dict[feature_list_name] :
    if vn not in train.columns:
        print "="*60 + vn
        
yv = train.click.values[filter_v1]

traintv_mx = train.as_matrix(feature_list_dict[feature_list_name])

print traintv_mx.shape


print "to save traintv_mx ..."

traintv_mx_save = {}
traintv_mx_save['traintv_mx'] = traintv_mx
traintv_mx_save['click'] = train.click.values
traintv_mx_save['day'] = train.day.values
traintv_mx_save['site_id'] = train.site_id.values
dump(traintv_mx_save, tmp_data_path + 'traintv_mx.joblib_dat')



