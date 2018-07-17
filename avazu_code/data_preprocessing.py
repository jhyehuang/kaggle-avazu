# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 17:45:28 2018

@author: huang
"""
import sys
import  csv
import time
import collections
import numpy as np
import pandas as pd

import logging
#sys.path.append('..')
#sys.path.append('D:/GitHub/jhye_tool')
from flags import FLAGS, unparsed
sys.path.append(FLAGS.tool_ml_dir)
from ml.ml_utils import *
from joblib import dump, load, Parallel, delayed
from sklearn.model_selection import train_test_split
import random
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.decomposition import PCA
import gc
import lightgbm as lgb 

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)


train_set_path = FLAGS.train_set_path
output = FLAGS.output_dir

def def_user(src_data):
    src_data['uid'] = src_data['device_id'].map(str).values + src_data['device_ip'].map(str).values + '-' + src_data['device_model'].map(str).values

def def_user_one_day_hour(src_data):
    src_data['uid_time'] = src_data['uid'].values + '-' + src_data['one_day_hour'].map(str).values


#  hour  shi jian tezheng
def anly_hour(src_data):
    src_data['date']=pd.to_datetime((src_data['hour'] / 100).map(int)+20000000,format='%Y%m%d')
    logging.debug(src_data['date'].unique())
    src_data['one_day']=src_data['date'].dt.day
    logging.debug(src_data['one_day'].unique())
    src_data['one_day_hour'] = src_data['date'].dt.hour
    src_data['week_day'] = src_data['date'].dt.dayofweek
    src_data['day_hour_prev']=src_data['one_day_hour']-1
    src_data['day_hour_next'] = src_data['one_day_hour'] + 1
    src_data['is_work_day'] = src_data['week_day'].apply(lambda x: 1 if x in [0,1,2,3,4] else 0)
#    src_data[src_data['is_work_day']==0]
    src_data.drop(['date'], axis=1,inplace = True)
    src_data.drop(['week_day'], axis=1,inplace = True)
    
    date_list=['one_day','one_day_hour','day_hour_prev','day_hour_next','is_work_day']
    
    src_data[date_list].to_csv(FLAGS.tmp_data_path+'date_list.csv',index=False)
            
def drop_limit_10(train,col_name):
    return dict((key,-1) if value <10 else (key,value)  for key,value in dict(train[col_name].value_counts()).items())

#  可以直接数的类别特征
def cat_features_cnt(src_data):
    id_cnt = collections.defaultdict(int)
    ip_cnt = collections.defaultdict(int)
    user_cnt = collections.defaultdict(int)
    user_hour_cnt = collections.defaultdict(int)
    id_cnt=drop_limit_10(src_data,'device_id')
    logging.debug(len(id_cnt))
    ip_cnt=drop_limit_10(src_data,'device_ip')
    logging.debug(len(ip_cnt))
    def_user(src_data)
    user_cnt=drop_limit_10(src_data,'uid')
    logging.debug(len(user_cnt))
    def_user_one_day_hour(src_data)
    user_hour_cnt=drop_limit_10(src_data,'uid_time')
    logging.debug(len(user_hour_cnt))
        
    return id_cnt,ip_cnt,user_cnt,user_hour_cnt


def col_one_hot(train):
    for _col in  train.columns.values.tolist():
        logging.debug(_col)
        if train[_col].dtypes=='object':
            ont=train[_col].astype('category').values.codes
            logging.debug(ont)
            train[_col]=ont
    return


FIELDS = ['C1','click','app_id','site_id','banner_pos','device_id','device_ip','device_model','device_conn_type','C14','C17','C20','C21']
#DATE_FIELDS=['one_day','date_time','day_hour_prev','one_day_hour','app_or_web','day_hour_next','app_site_id']
#NEW_FIELDS = FIELDS+DATE_FIELDS+['pub_id','pub_domain','pub_category','device_id_count','device_ip_count','user_count','smooth_user_hour_count','user_click_histroy']

#exptv_vn_list=['device_id','device_ip','C14','C17','C21',
#    'app_domain','site_domain','site_id','app_id','device_model','hour']

category_list = ['app_or_web',  'device_ip', 'app_site_id', 'device_model', 'app_site_model', 'C1', 'C14', 'C17', 'C21',
                            'device_type', 'device_conn_type','app_site_model_aw', 'dev_ip_app_site']
    
#exptv_vn_list=['C14','C17','C21','site_domain','device_model']
    

def add_col_cnt(src_data,col_name,cnt):
    vn=col_name+'_cnt'
    src_data[vn]=np.zeros(src_data.shape[0])
    func=lambda x: cnt[x]
    src_data[vn]=src_data[col_name].apply(func)
    logging.debug(src_data[vn].head())

# 可以在单条记录情况下 加工的类别特征
def one_line_data_preprocessing(is_train=True):
    src_data=pd.read_csv(FLAGS.tmp_data_path+'train_test.csv')
    anly_hour(src_data)
    logging.debug(src_data.shape)
    id_cnt,ip_cnt,user_cnt,user_hour_cnt=cat_features_cnt(src_data) 
    
    add_col_cnt(src_data,'device_id',id_cnt)
    add_col_cnt(src_data,'device_ip',ip_cnt)
    add_col_cnt(src_data,'uid',user_cnt)
    add_col_cnt(src_data,'uid_time',user_hour_cnt)
    col_one_hot(src_data)
    procdess_col(src_data,'app_id')
    procdess_col(src_data,'site_id')
    procdess_col(src_data,'app_domain')
    procdess_col(src_data,'app_category')
#    procdess_col(src_data,'site_id')
    num_writeheader_list=[]
    cat_writeheader_list=[]
    date_list=[]
    for col in src_data.columns.values.tolist():
        if col in category_list :
            cat_writeheader_list.append(col)
        elif 'day' in col:
            pass
#            date_list.append(col)
        else:
            num_writeheader_list.append(col)
    src_data[cat_writeheader_list].to_csv(FLAGS.tmp_data_path+'cat_features.csv',index=False)
#    src_data[date_list].to_csv(FLAGS.tmp_data_path+'date_list.csv',index=False)
    src_data[num_writeheader_list].to_csv(FLAGS.tmp_data_path+'num_features.csv',index=False)
    del src_data
    return 'cat_features.csv','date_list.csv','num_features.csv'
 


def two_features_data_preprocessing(data1,data2,data3, is_train=True):
    
    logging.debug("to add some basic features ...")
    
    #类别型特征俩俩 链接    
    calc_exptv(data1,data2,data3,category_list,add_count=True)
    new_expvn=calc_exptv_cnt()
    return  new_expvn

# 计算各特征的 权重
def new_features_w( is_train=True):
    
    data=pd.read_csv(FLAGS.tmp_data_path+'num_features.csv')
    new_expvn= ['C1', 'C15', 'C16', 'C18', 'C19', 'C20', ]
    src_data=data[new_expvn]
    src_data['click']=data['click'].values
    del data
    src_data=data_concat(src_data,FLAGS.tmp_data_path +'date_list.csv')
    #后验均值编码中的先验强度,随机给定强度
    n_ks={}
    for x in new_expvn:
        n_ks[x]=np.random.uniform(1, 500, 1)
    #初始化
    exp2_dict = {}
    for vn in new_expvn:
        exp2_dict[vn] = np.zeros(src_data.shape[0])
    days_npa = src_data.one_day.values
    
    for day_v in range(22, 32):
        # 将训练数据 分为 3 部分 : day_v之前 ,day_v
        day_v_before = src_data.ix[src_data.one_day.values < day_v, :].copy()
    
        #当前天的记录，作为校验集
        day_v_now = src_data.ix[src_data.one_day.values == day_v, :]
        logging.debug("Validation day:", day_v, ", train data shape:", day_v_before.shape, ", validation data shape:", day_v_now.shape)
    
        #初始化每个样本的y的先验 都等于 平均click率
        pred_prev = day_v_before.click.values.mean() * np.ones(day_v_before.shape[0])
    
        for vn in new_expvn:
            if 'exp2_'+vn in day_v_before.columns.values.tolist():  #已经有了，丢弃重新计算
                day_v_before.drop('exp2_'+vn, inplace=True, axis=1)
    
        for i in range(3):
            for vn in new_expvn:
                #计算对应的特征列中 在给定 y 的情况下的 概率
                p1 = calcLeaveOneOut2(day_v_before, vn, 'click', n_ks[vn], 0, 0.25, mean0=pred_prev)
                pred = pred_prev * p1
                logging.debug (day_v, i, vn, "change = ", ((pred - pred_prev)**2).mean())
                pred_prev = pred    
    
            #y的先验
            pred1 = day_v_before.click.values.mean()
            for vn in new_expvn:
                logging.debug("="*20, "merge", day_v, vn)
                diff1 = mergeLeaveOneOut2(day_v_before, day_v_now, vn)
                pred1 *= diff1
                exp2_dict[vn][days_npa == day_v] = diff1
            
            pred1 *= day_v_before.click.values.mean() / pred1.mean()
            logging.debug("logloss = ", logloss(pred1, day_v_now.click.values))

    t1=pd.DataFrame(np.zeros(src_data.shape[0]),columns=['click',])
    for vn in new_expvn:
        t1['exp2_'+vn] = exp2_dict[vn]
    t1.drop('click', axis=1,inplace = True)
    t1.to_csv(FLAGS.tmp_data_path+'new_features_w.csv',index=False)
    return 

#
def data_concat(src_data, dst_data_path,nrows=0,usecols=None, is_train=True):
    if usecols!=None:
        Reader_ = pd.read_csv(dst_data_path,usecols=[9,])
    elif nrows != 0:
        Reader_ = pd.read_csv(dst_data_path,nrows=nrows)
    else:
        Reader_ = pd.read_csv(dst_data_path,)
    try:
        Reader_.drop('id', axis=1,inplace = True)
    except:
        pass    
    
    logging.debug('data1.shape:'+str(src_data.shape))
    logging.debug('data2.shape:'+str(Reader_.shape))
    start = time.time()
    src_data=pd.concat([src_data,Reader_],axis = 1)
    logging.debug('结果.shape:'+str(src_data.shape))
    logging.debug('耗时'+str(time.time()-start))
#    return NEW_FIELDS
    return src_data



          

def concat_train_test(src_path, test_path,):
    train = pd.read_csv(src_path, dtype ={'id': object,})
    t5=pd.DataFrame(train['id'].map(int),columns=['id',])
#    logging.debug(t5.head(5))
#    t5.to_csv(FLAGS.tmp_data_path+'train_id.csv',index=False)
    col_cnts={}
    col_cnts['train']=(t5.shape[0])
    logging.debug(train.shape)
    del t5
    
    #训练集 乱序，下采样
    train = shuffle(train)
    train=train.sample(frac=0.15).reset_index(drop=True)
    
    
    test = pd.read_csv(test_path,dtype ={'id': object,})
    test['click'] = 0  #测试样本加一列click，初始化为0
    t6=pd.DataFrame(test['id'].map(str),columns=['id',])
    logging.debug(test['id'].map(str).head(5))
    logging.debug(t6.head(5))
    t6.to_csv(FLAGS.tmp_data_path+'test_id.csv',index=False)
    col_cnts['test']=(t6.shape[0])
    logging.debug(col_cnts)
    ret=dump(col_cnts, FLAGS.tmp_data_path+'test_index.joblib_dat')
    del t6

    
    logging.debug(test.shape)
    
#    try:
#        train.drop('id', axis=1,inplace = True)
#    except:
#        pass
#    try:
#        test.drop('id', axis=1,inplace = True)
#    except:
#        pass
    #将训练样本和测试样本连接，一起进行特征工程
    train = pd.concat([train, test])
    
    train['app_or_web'] = '0'
    #如果app_id='ecad2386',app_or_web=1
    train.ix[train.app_id.values=='ecad2386', 'app_or_web'] = '1'
    train['app_site_id'] = np.add(train.app_id.values, train.site_id.values)
    train['app_site_model'] = np.add(train.device_model.values, train.app_site_id.values)
    train['app_site_model_aw'] = np.add(train.app_site_model.values, train.app_or_web.values)
    train['dev_ip_app_site'] = np.add(train.device_ip.values, train.app_site_id.values)

    logging.debug(train.shape)
    
    
    train.to_csv(FLAGS.tmp_data_path+'train_test.csv',index=False)
    return 0


def features_by_chick():
    
    train_save = pd.read_csv(FLAGS.tmp_data_path +'cat_features.csv',)
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'date_list.csv')
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'click.csv')
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'two_col_join.csv')
    
    logging.debug(train_save['one_day'].unique())

    vns=[vn for vn in train_save.columns.values if 'day' not in vn ]
    #后验均值编码中的先验强度
    n_ks = {'app_or_web': 100, 'app_site_id': 100, 'device_ip': 10, 'C14': 50, 'app_site_model': 50, 'device_id': 50,
            'C17': 100, 'C21': 100, 'C1': 100, 'device_type': 100, 'device_conn_type': 100, 'banner_pos': 100,
            'app_site_model_aw': 100,'one_day':100, 'dev_ip_app_site': 10 , 'device_model': 500,'click':1}
    
#    vns=list(n_ks.keys())
    logging.debug(vns)
    logging.debug(train_save.one_day.unique())
    
    # 训练&测试
    train_save = train_save.ix[np.logical_and(train_save.one_day.values >= 21, train_save.one_day.values < 32), :]
    #串联两个特征成新的特征
    train_save['app_site_model'] = np.add(train_save.device_model.values, train_save.app_site_id.values)
    train_save['app_site_model_aw'] = np.add(train_save.app_site_model.values, train_save.app_or_web.values)
    train_save['dev_ip_app_site'] = np.add(train_save.device_ip.values, train_save.app_site_id.values)
    
    
    logging.debug(train_save.shape)
#    logging.debug(train_save.one_day.values)
    #初始化
    
    
    for vn in vns:
        if vn in n_ks:
            pass
        else:
            n_ks[vn]=100
    logging.debug (vn)
    

    #初始化
    exp2_dict = {}
    for vn in vns:
        exp2_dict[vn] = np.zeros(train_save.shape[0])
    
    days_npa = train_save.one_day.values
    logging.debug(days_npa)
        
    for day_v in range(22, 32):
        # day_v之前的天，所以从22开始，作为训练集
        logging.debug(train_save['one_day'])
        df1 = train_save.ix[np.logical_and(train_save.one_day.values < day_v, True), :].copy()
        logging.debug(df1.shape)
        #当前天的记录，作为校验集
        df2 = train_save.ix[train_save.one_day.values == day_v, :]
        logging.debug(df2.shape)
        print ("Validation day:", day_v, ", train data shape:", df1.shape, ", validation data shape:", df2.shape)
    
        #每个样本的y的先验都等于平均click率
        pred_prev = df1.click.values.mean() * np.ones(df1.shape[0])
    
    
        for vn in vns:
            if 'exp2_'+vn in df1.columns:  #已经有了，丢弃重新计算
                df1.drop('exp2_'+vn, inplace=True, axis=1)
    
        for i in range(3):
            for vn in vns:
                p1 = calcLeaveOneOut2(df1, vn, 'click', n_ks[vn], 0, 0.25, mean0=pred_prev)
                pred = pred_prev * p1
                print (day_v, i, vn, "change = ", ((pred - pred_prev)**2).mean())
                pred_prev = pred    
    
            #y的先验
            pred1 = df1.click.values.mean()
            for vn in vns:
                print ("="*20, "merge", day_v, vn)
                diff1 = mergeLeaveOneOut2(df1, df2, vn)
                pred1 *= diff1
                exp2_dict[vn][days_npa == day_v] = diff1
            
            pred1 *= df1.click.values.mean() / pred1.mean()
            print ("logloss = ", logloss(pred1, df2.click.values))
            #print my_lift(pred1, None, df2.click.values, None, 20, fig_size=(10, 5))
            #plt.show()
    exp_list=[]
    for vn in vns:
        train_save['exp2_'+vn] = exp2_dict[vn]
        exp_list.append('exp2_'+vn)
    train_save[exp_list].to_csv(FLAGS.tmp_data_path+'exp_features.csv',index=False)
    del train_save
    
def ouwenzhang():
    train_save = pd.read_csv(FLAGS.tmp_data_path +'cat_features.csv',)
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'date_list.csv')
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'click.csv')
    
    ori_col=train_save.columns.values
    print ("to count prev/current/next hour by ip ...")
    cntDualKey(train_save, 'device_ip', None, 'one_day', 'day_hour_prev', fill_na=0)
    cntDualKey(train_save, 'device_ip', None, 'one_day', 'one_day', fill_na=0)
    cntDualKey(train_save, 'device_ip', None, 'one_day', 'day_hour_next', fill_na=0)
    
    print( "to create day diffs")
    train_save['pday'] = train_save.one_day - 1
    calcDualKey(train_save, 'device_ip', None, 'one_day', 'pday', 'click', 10, None, True, True)
#    train_save['cnt_diff_device_ip_day_pday'] = train_save.cnt_device_ip_day.values  - train_save.cnt_device_ip_pday.values
    train_save['hour1_web'] = train_save.one_day_hour.values
    train_save.ix[train_save.app_or_web.values==0, 'hour1_web'] = -1
#    train_save['app_cnt_by_dev_ip'] = my_grp_cnt(train_save.device_ip.values, train_save.app_id.values)
    
    
    train_save['hour1'] = np.round(train_save.one_day_hour.values % 100)
#    train_save['cnt_diff_device_ip_day_pday'] = train_save.cnt_device_ip_day.values  - train_save.cnt_device_ip_pday.values
    
#    train_save['rank_dev_ip'] = my_grp_idx(train_save.device_ip.values, train_save.id.values)
    train_save['rank_day_dev_ip'] = my_grp_idx(np.add(train_save.device_ip.values, train_save.day.values), train_save.id.values)
#    train_save['rank_app_dev_ip'] = my_grp_idx(np.add(train_save.device_ip.values, train_save.app_id.values), train_save.id.values)
    
    
    train_save['cnt_dev_ip'] = get_agg(train_save.device_ip.values, train_save.id, np.size)
    train_save['cnt_dev_id'] = get_agg(train_save.device_id.values, train_save.id, np.size)
    
    train_save['dev_id_cnt2'] = np.minimum(train_save.cnt_dev_id.astype('int32').values, 300)
    train_save['dev_ip_cnt2'] = np.minimum(train_save.cnt_dev_ip.astype('int32').values, 300)
    
    train_save['dev_id2plus'] = train_save.device_id.values
    train_save.ix[train_save.cnt_dev_id.values == 1, 'dev_id2plus'] = '___only1'
    train_save['dev_ip2plus'] = train_save.device_ip.values
    train_save.ix[train_save.cnt_dev_ip.values == 1, 'dev_ip2plus'] = '___only1'
    
    train_save['diff_cnt_dev_ip_hour_phour_aw2_prev'] = (train_save.cnt_device_ip_day_hour.values - train_save.cnt_device_ip_day_hour_prev.values) * ((train_save.app_or_web * 2 - 1)) 
    train_save['diff_cnt_dev_ip_hour_phour_aw2_next'] = (train_save.cnt_device_ip_day_hour.values - train_save.cnt_device_ip_day_hour_next.values) * ((train_save.app_or_web * 2 - 1)) 
    
    now_col=train_save.columns.values
    new_col=[x for x in now_col if x not in ori_col]
    print("to save train_save ...")
    
    train_save[new_col].to_csv(FLAGS.tmp_data_path+'idx_features.csv',index=False)
    del train_save

def click_to_csv():
    num_features=pd.read_csv(FLAGS.tmp_data_path+'num_features.csv')
    t4=pd.DataFrame(num_features['click'].values,columns=['click',])
    t4.to_csv(FLAGS.tmp_data_path+'click.csv',index=False)
    del t4
    return True

def get_train_split():
    click=pd.read_csv(FLAGS.tmp_data_path+'click.csv')
    test_index = load(FLAGS.tmp_data_path+'test_index.joblib_dat')
    test_id=test_index['test']
    train_id=test_index['train']
    train_click=click[:train_id]
    filter_1 = np.logical_and(train_click.click.values > 0, True)
    filter_0 = np.logical_and(train_click.click.values == 0,True)
    files_name=['click.csv','cat_features.csv','date_list.csv','num_features.csv','two_col_join_cnt.csv','two_col_join.csv']
    
    
    logging.debug(files_name)
    for file in files_name:
        save=pd.read_csv(FLAGS.tmp_data_path+file)
        test_save=save[(-test_id):]
        test_save.to_csv(FLAGS.tmp_data_path+'test/'+file,index=False)
        logging.debug(test_save.shape)
        train_save=save[:train_id]
        for x in [100,299,799,1537]:
            train_0=train_save.ix[filter_0, :]
            train_1=train_save.ix[filter_1, :]
            prc=train_1.shape[0]/train_0.shape[0]
            train_1=train_1.sample(frac=0.5).reset_index(drop=True)
            logging.debug(train_1.shape)
            logging.debug(file)
            logging.debug(x )
            sampler = np.random.randint(0,train_0.shape[0],size=int(int(train_1.shape[0])/prc))
            train_0=train_0.take(sampler)
            train = pd.concat([train_0, train_1])
#            train = shuffle(train)
            train=train.sample(frac=1).reset_index(drop=True)
            logging.debug(train.shape)
            train.to_csv(FLAGS.tmp_data_path+'train'+str(x)+'/'+file,index=False)
            del train
            del train_0
            del train_1
            del sampler

def get_train_test_split():
    test_index = load(FLAGS.tmp_data_path+'test_index.joblib_dat')
    train_id=test_index['train']
#    test_id=test_index['test']

    files_name=['click.csv','cat_features.csv','date_list.csv','num_features.csv','two_col_join_cnt.csv','two_col_join.csv']
    
    
    logging.debug(files_name)
    for file in files_name:
        save=pd.read_csv(FLAGS.tmp_data_path+file)
#        test_save=save[(-test_id):]
#        test_save.to_csv(FLAGS.tmp_data_path+'test/'+file,index=False)
#        logging.debug(test_save.shape)
        train_save=save[:train_id]


#        train_save=train_save.sample(frac=0.005).reset_index(drop=True)
        logging.debug(train_save.shape)
        train_save.to_csv(FLAGS.tmp_data_path+'train25'+'/'+file,index=False)
        del train_save
        del save


def gdbt_data_get_train(seed=25):
    train_save = pd.read_csv(FLAGS.tmp_data_path +'train'+str(seed)+'/cat_features.csv',)
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'train'+str(seed)+'/date_list.csv')
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'train'+str(seed)+'/num_features.csv')
#    train_save=data_concat(train_save,FLAGS.tmp_data_path +'train100/click.csv')
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'train'+str(seed)+'/two_col_join.csv')
#    train_save=data_concat(train_save,FLAGS.tmp_data_path +'train'+str(seed)+'/two_col_join_cnt.csv')
    logging.debug(train_save.columns)

    logging.debug(train_save.shape)

    try:
        train_save.drop('id', axis=1,inplace = True)
    except:
        pass
    
    return train_save


def gdbt_data_get_test():
    test_save = pd.read_csv(FLAGS.tmp_data_path +'test/cat_features.csv',)
    test_save=data_concat(test_save,FLAGS.tmp_data_path +'test/date_list.csv')
    test_save=data_concat(test_save,FLAGS.tmp_data_path +'test/num_features.csv')
#    test_save=data_concat(test_save,FLAGS.tmp_data_path +'test/click.csv')
    test_save=data_concat(test_save,FLAGS.tmp_data_path +'test/two_col_join.csv')
#    test_save=data_concat(test_save,FLAGS.tmp_data_path +'test/two_col_join_cnt.csv')
    logging.debug(test_save.shape)

    try:
        test_save.drop('id', axis=1,inplace = True)
    except:
        pass
    
    
    test_save.drop('click',axis=1,inplace=True)
    return test_save


def lr_data_get(test_path):
    train_save = pd.read_csv(FLAGS.tmp_data_path +'train1537/cat_features.csv',)
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'train1537/date_list.csv')
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'train1537/num_features.csv')
#    train_save=data_concat(train_save,FLAGS.tmp_data_path +'train100/click.csv')
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'train1537/two_col_join.csv')
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'train1537/two_col_join_cnt.csv')
    logging.debug(train_save.columns)
#    logging.debug(train_save['id'])
    test_save = pd.read_csv(FLAGS.tmp_data_path +'test/cat_features.csv',)
    test_save=data_concat(test_save,FLAGS.tmp_data_path +'test/date_list.csv')
    test_save=data_concat(test_save,FLAGS.tmp_data_path +'test/num_features.csv')
#    test_save=data_concat(test_save,FLAGS.tmp_data_path +'test/click.csv')
    test_save=data_concat(test_save,FLAGS.tmp_data_path +'test/two_col_join.csv')
    test_save=data_concat(test_save,FLAGS.tmp_data_path +'test/two_col_join_cnt.csv')
    logging.debug(train_save.shape)
    logging.debug(test_save.shape)
    try:
        train_save.drop('id', axis=1,inplace = True)
    except:
        pass
    try:
        test_save.drop('id', axis=1,inplace = True)
    except:
        pass
    
    
    test_save.drop('click',axis=1,inplace=True)

    return train_save,test_save


def lightgbm_data_get(test_path):
    train_save = pd.read_csv(FLAGS.tmp_data_path +'train100/cat_features.csv',)
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'train100/date_list.csv')
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'train100/num_features.csv')
#    train_save=data_concat(train_save,FLAGS.tmp_data_path +'train100/click.csv')
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'train100/two_col_join.csv')
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'train100/two_col_join_cnt.csv')
    logging.debug(train_save.columns)
#    logging.debug(train_save['id'])
    test_save = pd.read_csv(FLAGS.tmp_data_path +'test/cat_features.csv',)
    test_save=data_concat(test_save,FLAGS.tmp_data_path +'test/date_list.csv')
    test_save=data_concat(test_save,FLAGS.tmp_data_path +'test/num_features.csv')
#    test_save=data_concat(test_save,FLAGS.tmp_data_path +'test/click.csv')
    test_save=data_concat(test_save,FLAGS.tmp_data_path +'test/two_col_join.csv')
    test_save=data_concat(test_save,FLAGS.tmp_data_path +'test/two_col_join_cnt.csv')
#    logging.debug(test_save.shape)
    logging.debug(train_save.shape)
    logging.debug(test_save.shape)
    try:
        train_save.drop('id', axis=1,inplace = True)
    except:
        pass
    try:
        test_save.drop('id', axis=1,inplace = True)
    except:
        pass
    
    print(train_save.shape)
    y_train = train_save['click']
    train_save.drop('click',axis=1,inplace=True)
    X_train = train_save
    
    test_save.drop('click',axis=1,inplace=True)
    X_test=test_save
    
    X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, y_train, train_size = 0.9,random_state = 0)
    logging.debug(X_train_part.head(1))
    logging.debug(y_train_part.head(1))
    ### 数据转换
    lgb_train = lgb.Dataset(X_train_part, y_train_part, free_raw_data=False)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train,free_raw_data=False)

    return lgb_train,lgb_eval,X_test,X_val,y_val

def tiny_lightgbm_data_get_train(seed=25):
    train_save = pd.read_csv(FLAGS.tmp_data_path + 'train'+str(seed)+'/cat_features.csv',)
    train_save=data_concat(train_save,FLAGS.tmp_data_path + 'train'+str(seed) +'/date_list.csv')
    train_save=data_concat(train_save,FLAGS.tmp_data_path + 'train'+str(seed) +'/num_features.csv')
#    train_save=data_concat(FLAGS.tmp_data_path + 'train'+str(seed) +'/click.csv')
    train_save=data_concat(train_save,FLAGS.tmp_data_path + 'train'+str(seed) +'/two_col_join.csv')
    train_save=data_concat(train_save,FLAGS.tmp_data_path + 'train'+str(seed) +'/two_col_join_cnt.csv')
    train_save=data_concat(train_save,FLAGS.tmp_data_path + 'train'+str(seed) +'/xgb_new_features.csv')
    logging.debug(train_save.columns)
#    logging.debug(train_save['id'])

#    logging.debug(test_save.shape)
    logging.debug(train_save.shape)
    try:
        train_save.drop('id', axis=1,inplace = True)
    except:
        pass

    
    print(train_save.shape)
    y_train = train_save['click']
    train_save.drop('click',axis=1,inplace=True)
    X_train = train_save
    
    
    X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, y_train, train_size = 0.9,random_state = 0)
    logging.debug(X_train_part.head(1))
    logging.debug(y_train_part.head(1))
    ### 数据转换
    lgb_train = lgb.Dataset(X_train_part, y_train_part, free_raw_data=False)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train,free_raw_data=False)

    return lgb_train,lgb_eval,X_val,y_val

def tiny_lightgbm_data_get_test():

    test_save = pd.read_csv(FLAGS.test_data_path +'cat_features.csv',)
    test_save=data_concat(test_save,FLAGS.test_data_path +'date_list.csv')
    test_save=data_concat(test_save,FLAGS.test_data_path +'num_features.csv')
#    test_save=data_concat(FLAGS.tmp_data_path +'test/click.csv')
    test_save=data_concat(test_save,FLAGS.test_data_path +'two_col_join.csv')
    test_save=data_concat(test_save,FLAGS.test_data_path +'two_col_join_cnt.csv')
    test_save=data_concat(test_save,FLAGS.test_data_path +'xgb_new_features.csv')
    logging.debug(test_save.shape)

    try:
        test_save.drop('id', axis=1,inplace = True)
    except:
        pass
    
    test_save.drop('click',axis=1,inplace=True)
    

    return test_save


def pandas_onehot(df, col):
    df = pd.get_dummies(df, columns=col)
    return df

def sklearn_onehoot(df,col):
    enc = OneHotEncoder()
    enc.fit(df)  
    data = enc.transform(df).toarray()
    return data

columns_all=['C14', 'C17', 'C21', 'device_model', 'site_domain',
             'C1', 'C15', 'C16', 'C18', 'C19', 'C20', 'app_category', 
             'app_domain', 'app_id', 'banner_pos', 'device_conn_type', 
             'device_id', 'device_ip', 'device_type', 'hour', 'site_category', 
             'site_id', 'uid', 'uid_time', 'device_id_cnt', 'device_ip_cnt', 
             'uid_cnt', 'uid_time_cnt', 'C14C17', '_key1', 'C14device_model', 
             'C14C21', 'C14site_domain', 'C17device_model', 'C17C21', 'C17site_domain',
             'C21device_model', 'C21site_domain', 'site_domaindevice_model', 'cnttv_C14C17',
             'cnttv_C14device_model', 'cnttv_C14C21', 'cnttv_C14site_domain', 'cnttv_C17device_model', 
             'cnttv_C17C21', 'cnttv_C17site_domain', 'cnttv_C21device_model', 'cnttv_C21site_domain',
             'cnttv_site_domaindevice_model']

columns_top=['site_id', 'hour', 'app_id', 'C19', 'device_ip_cnt', 'C20', 'site_category', 
             'uid_cnt', 'app_domain', 'device_id_cnt', 'device_ip', 'C18', 'uid_time_cnt', 
             'device_model', 'app_category', 'site_domain', 'C21device_model', 
             'exptv_site_domaindevice_model', 'banner_pos', 'C14', 'exptv_C21device_model',
             'cnttv_C21device_model', 'cnttv_site_domaindevice_model', '_key1', 'cnttv_C14device_model',
             'exptv_C14device_model', 'C16', 'cnttv_C17device_model', 'exptv_C17device_model', 
             'device_conn_type', 'device_id', 'cnttv_C14site_domain', 'cnttv_C17C21', 
             'exptv_C21site_domain', 'cnttv_C21site_domain', 'C17', 'C21', 'uid', 'C17device_model',
             'cnttv_C17site_domain', 'exptv_C14site_domain', 'site_domaindevice_model', 'C21site_domain',
             'exptv_C17site_domain', 'top_1_site_id', 'cnttv_C14C17', 'exptv_C17C21', 'C1', 
             'C14device_model', 'top_2_site_id', 'uid_time', 'top_5_site_id', 'C15', 'exptv_C14C17',
             'C14site_domain', 'top_1_app_id', 'cnttv_C14C21', 'C17site_domain', 'top_2_app_id', 
             'device_type', 'top_10_site_id', 'exptv_C14C21']
columns_100002w=['device_id', 'device_ip', 'device_id_cnt', 'device_ip_cnt', '_key1', 
                'C14device_model', 'C17device_model', 'site_domaindevice_model', 
                'cnttv_C14C17', 'cnttv_C14C21', 'cnttv_C14site_domain', 'cnttv_C17device_model', 
                'cnttv_C17C21', 'cnttv_C17site_domain', 'cnttv_C21device_model', 'cnttv_C21site_domain']

columns = [item for item in columns_top if item not in columns_100002w]
def col_one_hot2(train,one_field):
#    enc = OneHotEncoder()
#    enc.fit(train)
    
    logging.debug(train.head(2))
    logging.debug(one_field)
    now = time.time()
    logging.debug('Format Converting begin in time:...')
    logging.debug(now)
    columns = train.columns.values

    d = len(columns)
    feature_index = [i for i in range(d)]
    field_index = [0]*d
    field = []
    for col in columns:
        field.append(col)
    index = -1
    for i in range(d):
        if i==0 or field[i]!=field[i-1]:
            index+=1
        field_index[i] = index

    fp=FLAGS.tmp_data_path +one_field+'-ont_hot_train.libffm.txt'
    with open(fp, 'w') as f:
        for row_no,row in enumerate(train.values):
            line =str(row_no)          
#            row= enc.transform(row).toarray()
            logging.debug(row)
            for i in range(1, len(row)):
                if row[i]!=0:
                    line += ' ' + "%s:%d:%d:" % (one_field,train.values, 1) + ' '
            line+='\n'
            f.write(line)
    logging.debug('finish convert,the cost time is ')
    logging.debug(time.time()-now)
    logging.debug('[Done]')
    logging.debug()
#    return  pd.DataFrame(train)


def features_index():
    pass

def train_data_ont_hot(seed=25):
    train_save = pd.read_csv(FLAGS.tmp_data_path + 'train'+str(seed)+'/click.csv',)
#    train_save=data_concat(train_save,FLAGS.tmp_data_path + 'train'+str(seed) +'/date_list.csv')
#    train_save=data_concat(train_save,FLAGS.tmp_data_path + 'train'+str(seed) +'/num_features.csv')
#    train_save=data_concat(train_save,FLAGS.tmp_data_path + 'train'+str(seed) +'/cat_features.csv')
#    train_save=data_concat(train_save,FLAGS.tmp_data_path + 'train'+str(seed) +'/two_col_join.csv')
#    train_save=data_concat(train_save,FLAGS.tmp_data_path + 'train'+str(seed) +'/two_col_join_cnt.csv')
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'train'+str(seed) +'/xgb_new_features.csv')
    logging.debug(train_save.columns)
#    logging.debug(train_save['id'])

#    logging.debug(test_save.shape)
    logging.debug(train_save.shape)
    try:
        train_save.drop('id', axis=1,inplace = True)
    except:
        pass

    
    logging.debug(train_save.shape)
    try:
        y_train = train_save['click']
        train_save.drop('click',axis=1,inplace=True)
    except:
        pass    
    columns=train_save.columns.values
    train_save=train_save[columns]
    features = list(train_save.columns)
    for feature_index,feature in enumerate(features):
        def set_field_feature_value(row):
            return "%d:%d:%d" % (feature_index,row, 1)
        logging.debug(feature+':cnt:'+str(train_save[feature].max()))
        now=time.time()
        logging.debug(feature + ' Format Converting begin in time:...')
        logging.debug(now)
        max_ = train_save[feature].max()
        train_save[feature] = (train_save[feature] - max_) * (-1)
        train_save[feature]=train_save[feature].apply(set_field_feature_value)
#        train_save['label']=y_train
        logging.debug(feature + ' finish convert,the cost time is ')
        logging.debug(time.time()-now)
#        one_col=pandas_onehot(train_save.loc[:,feature],feature)
#        logging.debug(one_col.shape)
#        col_one_hot(one_col,feature)
#        del one_col
    fp=FLAGS.tmp_data_path +'ont_hot_train.libffm.csv'
    now=time.time()
#    train_save=pd.concat([y_train,train_save],axis = 1)
    logging.debug(time.time()-now)
#    with open(fp, 'w') as f:
#        for y,row in zip(y_train.values,train_save.values):
#            logging.debug(row)
#            row=[str(x) for x in row]
#            line=str(y)+' '+' '.join(row)+'\n'
#            f.write(line)
    train_save.to_csv(fp, sep=' ', header=False, index=False)
    logging.debug('finish convert,the cost time is ')
    logging.debug(time.time()-now)
    logging.debug('[Done]')
    
    logging.debug(train_save.head(2))
    logging.debug(train_save.shape)
    del train_save

def vali_data_ont_hot(seed=799):
    train_save = pd.read_csv(FLAGS.tmp_data_path + 'train'+str(seed)+'/click.csv',)
    train_save=data_concat(train_save,FLAGS.tmp_data_path + 'train'+str(seed) +'/date_list.csv')
#    train_save=data_concat(train_save,FLAGS.tmp_data_path + 'train'+str(seed) +'/num_features.csv')
#    train_save=data_concat(train_save,FLAGS.tmp_data_path + 'train'+str(seed) +'/cat_features.csv')
#    train_save=data_concat(train_save,FLAGS.tmp_data_path + 'train'+str(seed) +'/two_col_join.csv')
#    train_save=data_concat(train_save,FLAGS.tmp_data_path + 'train'+str(seed) +'/two_col_join_cnt.csv')
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'train'+str(seed) +'/xgb_new_features.csv')
    logging.debug(train_save.columns)
#    logging.debug(train_save['id'])

#    logging.debug(test_save.shape)
    logging.debug(train_save.shape)
    try:
        train_save.drop('id', axis=1,inplace = True)
    except:
        pass

    
    logging.debug(train_save.shape)
    try:
        y_train = train_save['click']
        train_save.drop('click',axis=1,inplace=True)
    except:
        pass    
    columns=train_save.columns.values
    train_save=train_save[columns]
    features = list(train_save.columns)
    for feature_index,feature in enumerate(features):
        def set_field_feature_value(row):
            return "%d:%d:%d" % (feature_index,row, 1)
        now=time.time()
        logging.debug(feature + ' Format Converting begin in time:...')
        logging.debug(now)
        max_ = train_save[feature].max()
        train_save[feature] = (train_save[feature] - max_) * (-1)
        train_save[feature]=train_save[feature].apply(set_field_feature_value)
#        train_save['label']=y_train
        logging.debug(feature + ' finish convert,the cost time is ')
        logging.debug(time.time()-now)
#        one_col=pandas_onehot(train_save.loc[:,feature],feature)
#        logging.debug(one_col.shape)
#        col_one_hot(one_col,feature)
#        del one_col
    fp=FLAGS.tmp_data_path +'ont_hot_vali.libffm.csv'
    train_save=pd.concat([y_train,train_save],axis = 1)
#    with open(fp, 'w') as f:
#        for y,row in zip(y_train.values,train_save.values):
#            logging.debug(row)
#            row=[str(x) for x in row]
#            line=str(y)+' '+' '.join(row)+'\n'
#            f.write(line)
    train_save.to_csv(fp, sep=' ', header=False, index=False)
    logging.debug('finish convert,the cost time is ')
    logging.debug(time.time()-now)
    logging.debug('[Done]')
    
    logging.debug(train_save.head(2))
    logging.debug(train_save.shape)
    del train_save

def test_data_ont_hot():
    test_save = pd.read_csv(FLAGS.tmp_data_path +'test/click.csv',)
    test_save=data_concat(test_save,FLAGS.tmp_data_path +'test/date_list.csv')
#    test_save=data_concat(test_save,FLAGS.tmp_data_path +'test/num_features.csv')
#    test_save=data_concat(test_save,FLAGS.tmp_data_path +'test/cat_features.csv')
#    test_save=data_concat(test_save,FLAGS.tmp_data_path +'test/two_col_join.csv')
#    test_save=data_concat(test_save,FLAGS.tmp_data_path +'test/two_col_join_cnt.csv')
    test_save=data_concat(test_save,FLAGS.tmp_data_path +'test/xgb_new_features.csv')
    logging.debug(test_save.shape)

    try:
        test_save.drop('id', axis=1,inplace = True)
    except:
        pass
    logging.debug(test_save.columns)
#    logging.debug(train_save['id'])
    
    logging.debug(test_save.shape)
    try:
        y_train = test_save['click']
        test_save.drop('click',axis=1,inplace=True)
    except:
        pass   
    columns=test_save.columns.values
    test_save=test_save[columns]
    features = list(test_save.columns)
    for feature_index,feature in enumerate(features):
        def set_field_feature_value(row):
            return "%d:%d:%d" % (feature_index,row, 1)
        now=time.time()
        logging.debug(feature + ' Format Converting begin in time:...')
        logging.debug(now)
        max_ = test_save[feature].max()
        test_save[feature] = (test_save[feature] - max_) * (-1)
        test_save[feature]=test_save[feature].apply(set_field_feature_value)
#        train_save['label']=y_train
        logging.debug(feature + ' finish convert,the cost time is ')
        logging.debug(time.time()-now)
#        one_col=pandas_onehot(train_save.loc[:,feature],feature)
#        logging.debug(one_col.shape)
#        col_one_hot(one_col,feature)
#        del one_col
    fp=FLAGS.tmp_data_path +'ont_hot_test.libffm.csv'
    train_save=pd.concat([y_train,test_save],axis = 1)
#    with open(fp, 'w') as f:
#        for y,row in zip(y_train.values,train_save.values):
#            logging.debug(row)
#            row=[str(x) for x in row]
#            line=str(y)+' '+' '.join(row)+'\n'
#            f.write(line)
    train_save.to_csv(fp, sep=' ', header=False, index=False)
    logging.debug('finish convert,the cost time is ')
    logging.debug(time.time()-now)
    logging.debug('[Done]')
    
    logging.debug(train_save.head(2))
    logging.debug(train_save.shape)
    del train_save
    del test_save

def gdbt_DM_get_train(seed=25):
    train_save = pd.read_csv(FLAGS.tmp_data_path +'train'+str(seed)+'/cat_features.csv',)
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'train'+str(seed)+'/date_list.csv')
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'train'+str(seed)+'/num_features.csv')
#    train_save=data_concat(train_save,FLAGS.tmp_data_path +'train100/click.csv')
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'train'+str(seed)+'/two_col_join.csv')
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'train'+str(seed)+'/two_col_join_cnt.csv')
    logging.debug(train_save.columns)

    logging.debug(train_save.shape)

    
    try:
        train_save.drop('id', axis=1,inplace = True)
    except:
        pass
    
    y_train = train_save['click']
    train_save.drop('click',axis=1,inplace=True)
    pca = PCA(n_components=0.85)
      
    
    X_train_part, X_val, y_train_part, y_val = train_test_split(train_save, y_train, train_size = 0.6,random_state = 7)
    pca.fit(X_train_part[:200000])
    X_train_part=pca.transform(X_train_part)
    dtrain = xgb.DMatrix(X_train_part, label=y_train_part)
    dtrain.save_binary(FLAGS.tmp_data_path+'train'+str(seed)+'/xgboost.new_features.dtrain.joblib_dat')
    del dtrain,X_train_part,y_train_part
    gc.collect()

    X_val=pca.transform(X_val)
    dvalid = xgb.DMatrix(X_val, label=y_val)
    dvalid.save_binary(FLAGS.tmp_data_path+'train'+str(seed)+'/xgboost.new_features.dvalid.joblib_dat')
    del dvalid,X_val,y_val
    gc.collect()

    train_save=pca.transform(train_save)
    dtv = xgb.DMatrix(train_save)
    dtv.save_binary(FLAGS.tmp_data_path+'train'+str(seed)+'/xgboost.new_features.dtv.joblib_dat')
    del dtv,train_save
    gc.collect()

    return 0

def gdbt_DM_get_test():
    test_save = pd.read_csv(FLAGS.tmp_data_path +'test/cat_features.csv',)
    test_save=data_concat(test_save,FLAGS.tmp_data_path +'test/date_list.csv')
    test_save=data_concat(test_save,FLAGS.tmp_data_path +'test/num_features.csv')
#    test_save=data_concat(test_save,FLAGS.tmp_data_path +'test/click.csv')
    test_save=data_concat(test_save,FLAGS.tmp_data_path +'test/two_col_join.csv')
    test_save=data_concat(test_save,FLAGS.tmp_data_path +'test/two_col_join_cnt.csv')
    logging.debug(test_save.shape)

    try:
        test_save.drop('id', axis=1,inplace = True)
    except:
        pass
    
    
    test_save.drop('click',axis=1,inplace=True)
    dtv = xgb.DMatrix(test_save)
    dtv.save_binary(FLAGS.tmp_data_path+'test/xgboost.new_features.test.joblib_dat')
    del dtv,test_save
    gc.collect()
    return 0

def get_PCA_train_data(seed=25):
    train_save = pd.read_csv(FLAGS.tmp_data_path +'train'+str(seed)+'/cat_features.csv',nrows=200000)
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'train'+str(seed)+'/date_list.csv',nrows=200000)
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'train'+str(seed)+'/num_features.csv',nrows=200000)
#    train_save=data_concat(train_save,FLAGS.tmp_data_path +'train100/click.csv',nrows=200000)
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'train'+str(seed)+'/two_col_join.csv',nrows=200000)
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'train'+str(seed)+'/two_col_join_cnt.csv',nrows=200000)
    logging.debug(train_save.columns)

    logging.debug(train_save.shape)
    try:
        train_save.drop('id', axis=1,inplace = True)
    except:
        pass
    
    
    gc.collect()

    return train_save
