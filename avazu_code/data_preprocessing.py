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

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)


train_set_path = FLAGS.train_set_path
output = FLAGS.output_dir

def def_user(row):
    user = row['device_id'] + row['device_ip'] + '-' + row['device_model']
    return user

#  可以直接数的类别特征
def cat_features_cnt(path):
    id_cnt = collections.defaultdict(int)
    ip_cnt = collections.defaultdict(int)
    user_cnt = collections.defaultdict(int)
    user_hour_cnt = collections.defaultdict(int)
    for i, row in enumerate(csv.DictReader(open(path)), start=1):
        start = time.time()
        if i % 1000000 == 0:
            sys.stderr.write('{0}s    {1}mil\n'.format((time.time()-start),int(i/1000000)))

        user = def_user(row)
        id_cnt[row['device_id']] += 1
        ip_cnt[row['device_ip']] += 1
        user_cnt[user] += 1
        user_hour_cnt[user+'-'+row['hour']] += 1
        
    return id_cnt,ip_cnt,user_cnt,user_hour_cnt

FIELDS = ['id','click','hour','app_id','site_id','banner_pos','device_id','device_ip','device_model','device_conn_type','C14','C17','C20','C21']
DATE_FIELDS=['one_day','date_time','day_hour_prev','one_day_hour','app_or_web','day_hour_next','app_site_id']
NEW_FIELDS = FIELDS+DATE_FIELDS+['pub_id','pub_domain','pub_category','device_id_count','device_ip_count','user_count','smooth_user_hour_count','user_click_histroy']

exptv_vn_list=['device_id','device_ip','C14','C17','C21',
    'app_domain','site_domain','site_id','app_id','device_model','hour']
    
# 可以在单条记录情况下 加工的类别特征
def one_line_data_preprocessing(src_path, dst_app_path, is_train=True):
    id_cnt,ip_cnt,user_cnt,user_hour_cnt=cat_features_cnt(src_path) 
    reader = csv.DictReader(open(src_path))
    writer_app = csv.DictWriter(open(dst_app_path, 'w'), NEW_FIELDS)
    writer_app.writeheader()
    start = time.time()
    for i, row in enumerate(reader, start=1):
        if i % 1000000 == 0:
            sys.stderr.write('{0}s    {1}mil\n'.format((time.time()-start),int(i/1000000)))
        
        new_row = {}
        for field in FIELDS:
            new_row[field] = row[field]

        new_row['device_id_count'] = id_cnt[row['device_id']]
        new_row['device_ip_count'] = ip_cnt[row['device_ip']]

        user, hour = def_user(row), row['hour']
        new_row['user_count'] = user_cnt[user]
        new_row['smooth_user_hour_count'] = str(user_hour_cnt[user+'-'+hour])
        

        new_row['one_day']=int(new_row['hour']) % 10000 / 100
        new_row['one_day_hour'] = int(new_row['hour'])%100
#        new_row['week_day'] = new_row['date'].dt.dayofweek
        new_row['date_time'] = (new_row['one_day'] - 21) * 24 + new_row['one_day_hour']
        new_row['day_hour_prev'] = new_row['one_day_hour'] - 1
        new_row['day_hour_next'] = new_row['one_day_hour'] + 1
#        new_row['is_work_day'] = new_row['week_day'].apply(lambda x: 1 if x in [0,1,2,3,4] else 0)
#        new_row.drop(['date'], axis=1,inplace = True)
        new_row['app_or_web']= 1 if new_row['app_id']=='ecad2386' else 0
        new_row['app_site_id'] = new_row['app_id']+new_row['site_id']


        if True if row['site_id'] == '85f751fd' else False:
            new_row['pub_id'] = row['app_id']
            new_row['pub_domain'] = row['app_domain']
            new_row['pub_category'] = row['app_category']
            
        else:
            new_row['pub_id'] = row['site_id']
            new_row['pub_domain'] = row['site_domain']
            new_row['pub_category'] = row['site_category']
        writer_app.writerow(new_row)
    return NEW_FIELDS

def two_features_data_preprocessing(path, is_train=True):
    
    logging.debug("to add some basic features ...")
    
    #类别型特征俩俩 链接    
    
    new_expvn=calc_exptv(path, exptv_vn_list,add_count=True)
    return  new_expvn

# 计算各特征的 权重
def new_features_w(src_data, new_expvn, is_train=True):
    
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
            if 'exp2_'+vn in day_v_before.columns:  #已经有了，丢弃重新计算
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

    for vn in new_expvn:
        src_data['exp2_'+vn] = exp2_dict[vn]

#
def data_concat(src_data, dst_app_path, dst_site_path, is_train=True):

    Reader_app = pd.read_csv(open(dst_app_path, "r"))
    Reader_site = pd.read_csv(open(dst_site_path, "r"))
    
    src_data= pd.merge(src_data, Reader_app, on=FIELDS)
    src_data= pd.merge(src_data, Reader_site, on=FIELDS)
    start = time.time()
    logging.debug(time.time()-start)
    return NEW_FIELDS

def drop_outpoint(col_name_list,tmp_data_path):
    for col_name in col_name_list:
        train_one = pd.read_csv(tmp_data_path+col_name)
        t1=check_col_count_less_11(train_one[col_name],col_name)
        train_one[train_one[col_name]==t1.keys()]=-1
        train_one.to_csv(tmp_data_path+col_name)

def data_to_col_csv(col_name_list,src_train_path, tmp_data_path):
    writeheader_list=[]
    for col in col_name_list:
        obj_name=''
        columns=[]
        columns.append(col)
        obj_name= csv.DictWriter(open(tmp_data_path+col, 'w'), columns)
        obj_name.writeheader()
        writeheader_list.append(obj_name)

    reader = csv.DictReader(open(src_train_path))
    start=time.time()
    for i, row in enumerate(reader, start=1):
        if i % 1000000 == 0:
            sys.stderr.write('{0}s    {1}mil\n'.format((time.time()-start),int(i/1000000)))

        for field,col_writeheader in zip(col_name_list,writeheader_list):
            #print(field)
            #print(col_writeheader)
            new_row={}
            new_row[field]=row[field]
            col_writeheader.writerow(new_row)
    drop_outpoint(col_name_list, tmp_data_path)
            

def concat_train_test(src_path, dst_app_path,):
    _fields=['id','click', 'hour', 'C1', 'banner_pos' ,'site_id', 'site_domain',
     'site_category', 'app_id', 'app_domain' ,'app_category', 'device_id',
     'device_ip' ,'device_model', 'device_type', 'device_conn_type', 'C14', 'C15',
     'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
    reader = csv.DictReader(open(src_path))
    writer_app = csv.DictWriter(open(dst_app_path, 'w+'), _fields)
    writer_app.writeheader()
    start = time.time()
    for i, row in enumerate(reader, start=1):
        if i % 1000000 == 0:
            sys.stderr.write('{0}s    {1}mil\n'.format((time.time()-start),int(i/1000000)))
        
        new_row = {}
        row['click']=0
        for field in _fields:
            new_row[field] = row[field]
        print(new_row)
        writer_app.writerow(new_row)
    return True