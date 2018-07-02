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
    src_data['date']=pd.to_datetime((src_data['hour'] / 100).map(int)+20000000)
    src_data['one_day']=src_data['date'].dt.day
    src_data['one_day_hour'] = src_data['date'].dt.hour
    src_data['week_day'] = src_data['date'].dt.dayofweek
    src_data['day_hour_prev']=src_data['one_day_hour']-1
    src_data['day_hour_next'] = src_data['one_day_hour'] + 1
    src_data['is_work_day'] = src_data['week_day'].apply(lambda x: 1 if x in [0,1,2,3,4] else 0)
#    src_data[src_data['is_work_day']==0]
    src_data.drop(['date'], axis=1,inplace = True)
    src_data.drop(['week_day'], axis=1,inplace = True)
            
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

exptv_vn_list=['C14','C17','C21','site_domain','device_model']
    

def add_col_cnt(src_data,col_name,cnt):
    vn=col_name+'_cnt'
    src_data[vn]=np.zeros(src_data.shape[0])
    func=lambda x: cnt[x]
    src_data[vn]=src_data[col_name].apply(func)
    logging.debug(src_data[vn].head())

# 可以在单条记录情况下 加工的类别特征
def one_line_data_preprocessing(src_data, dst_app_path, is_train=True):
    
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

    return src_data.columns.values.tolist()

def two_features_data_preprocessing(data1,data2,data3, is_train=True):
    
    logging.debug("to add some basic features ...")
    
    #类别型特征俩俩 链接    
    calc_exptv(data1,data2,data3,exptv_vn_list,add_count=True)
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
def data_concat(src_data, dst_data_path,usecols=None, is_train=True):
    if usecols!=None:
        Reader_ = pd.read_csv(dst_data_path,usecols=[9,])
    else:
        Reader_ = pd.read_csv(dst_data_path)
    try:
        Reader_.drop('id', axis=1,inplace = True)
    except:
        pass    
    
    logging.debug('data1.shape:'+str(src_data.shape))
    logging.debug('data2.shape:'+str(Reader_.shape))
    src_data=pd.concat([src_data,Reader_],axis = 1)
    start = time.time()
    logging.debug('结果.shape:'+str(src_data.shape))
    logging.debug('耗时'+str(time.time()-start))
#    return NEW_FIELDS
    return src_data



def data_to_col_csv(col_name_list,train, tmp_data_path):
    num_writeheader_list=[]
    cat_writeheader_list=[]
    date_list=[]
    for col in col_name_list:
        if col in exptv_vn_list :
            cat_writeheader_list.append(col)
        elif 'day' in col:
            date_list.append(col)
        else:
            num_writeheader_list.append(col)
    train[cat_writeheader_list].to_csv(FLAGS.tmp_data_path+'cat_features.csv',index=False)
    train[date_list].to_csv(FLAGS.tmp_data_path+'date_list.csv',index=False)
    train[num_writeheader_list].to_csv(FLAGS.tmp_data_path+'num_features.csv',index=False)
    del train
    return 'cat_features.csv','date_list.csv','num_features.csv'
            

def concat_train_test(src_path, test_path,):
    train = pd.read_csv(src_path, dtype ={'id': object,})
    t5=pd.DataFrame(train['id'].map(int),columns=['id',])
#    logging.debug(t5.head(5))
    t5.to_csv(FLAGS.tmp_data_path+'train_id.csv',index=False)
    col_cnts={}
    col_cnts['train']=(t5.shape[0])
    logging.debug(train.shape)
    del t5
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
    
    try:
        train.drop('id', axis=1,inplace = True)
    except:
        pass
    try:
        test.drop('id', axis=1,inplace = True)
    except:
        pass
    #将训练样本和测试样本连接，一起进行特征工程
    train = pd.concat([train, test])
    logging.debug(train.shape)
#    try:
#        train.drop('id', axis=1,inplace = True)
#    except:
#        pass
    return train

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
    files_name=['click.csv','cat_features.csv','date_list.csv','id.csv','num_features.csv','two_col_join_cnt.csv','two_col_join.csv']
    
    
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
            train=train.sample(frac=0.5).reset_index(drop=True)
            logging.debug(train.shape)
            train.to_csv(FLAGS.tmp_data_path+'train'+str(x)+'/'+file,index=False)
            del train
            del train_0
            del train_1
            del sampler

def gdbt_data_get_train(seed=1537):
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
    
    return train_save


def gdbt_data_get_test():
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

def tiny_lightgbm_data_get_train():
    train_save = pd.read_csv(FLAGS.tmp_data_path +'cat_features.csv',)
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'date_list.csv')
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'num_features.csv')
#    train_save=data_concat(train_save,FLAGS.tmp_data_path +'click.csv')
    train_save=data_concat(train_save,FLAGS.tmp_data_path +'two_col_join.csv')
#    train_save=data_concat(train_save,FLAGS.tmp_data_path +'two_col_join_cnt.csv')
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
#    test_save=data_concat(test_save,FLAGS.test_data_path +'click.csv')
    test_save=data_concat(test_save,FLAGS.test_data_path +'two_col_join.csv')
#    test_save=data_concat(test_save,FLAGS.test_data_path +'two_col_join_cnt.csv')
    logging.debug(test_save.shape)

    try:
        test_save.drop('id', axis=1,inplace = True)
    except:
        pass
    
    test_save.drop('click',axis=1,inplace=True)
    

    return test_save

