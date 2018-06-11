# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 17:45:28 2018

@author: huang
"""
import sys
import  csv
import time
import collections

def def_user(row):
    user = row['device_id'] + row['device_ip'] + '-' + row['device_model']
    return user

def cat_features_cnt(path):
    id_cnt = collections.defaultdict(int)
    ip_cnt = collections.defaultdict(int)
    user_cnt = collections.defaultdict(int)
    user_hour_cnt = collections.defaultdict(int)
    for i, row in enumerate(csv.DictReader(open(path)), start=1):
        start = time.time()
        if i % 1000000 == 0:
            sys.stderr.write('{0:6.0f}    {1}m\n'.format(time.time()-start,int(i/1000000)))

        user = def_user(row)
        id_cnt[row['device_id']] += 1
        ip_cnt[row['device_ip']] += 1
        user_cnt[user] += 1
        user_hour_cnt[user+'-'+row['hour']] += 1
        
    return id_cnt,ip_cnt,user_cnt,user_hour_cnt

FIELDS = ['id','click','hour','banner_pos','device_id','device_ip','device_model','device_conn_type','C14','C17','C20','C21']
NEW_FIELDS = FIELDS+['pub_id','pub_domain','pub_category','device_id_count','device_ip_count','user_count','smooth_user_hour_count','user_click_histroy']

def one_line_data_preprocessing(src_path, dst_app_path, dst_site_path, is_train):
    id_cnt,ip_cnt,user_cnt,user_hour_cnt=cat_features_cnt(src_path) 
    reader = csv.DictReader(open(src_path))
    writer_app = csv.DictWriter(open(dst_app_path, 'w'), NEW_FIELDS)
    writer_site = csv.DictWriter(open(dst_site_path, 'w'), NEW_FIELDS)
    writer_app.writeheader()
    writer_site.writeheader()
    start = time.time()
    for i, row in enumerate(reader, start=1):
        if i % 1000000 == 0:
            sys.stderr.write('{0:6.0f}    {1}m\n'.format(time.time()-start,int(i/1000000)))
        
        new_row = {}
        for field in FIELDS:
            new_row[field] = row[field]

        new_row['device_id_count'] = id_cnt[row['device_id']]
        new_row['device_ip_count'] = ip_cnt[row['device_ip']]

        user, hour = def_user(row), row['hour']
        new_row['user_count'] = user_cnt[user]
        new_row['smooth_user_hour_count'] = str(user_hour_cnt[user+'-'+hour])
        
        new_row['day']=new_row['hour'] % 10000 / 100
        new_row['hour1'] = new_row['hour'] % 100
        new_row['day_hour'] = (new_row['day'] - 21) * 24 + new_row['hour1']
        new_row['day_hour_prev'] = new_row['day_hour'] - 1
        new_row['day_hour_next'] = new_row['day_hour'] + 1


        if True if row['site_id'] == '85f751fd' else False:
            new_row['pub_id'] = row['app_id']
            new_row['pub_domain'] = row['app_domain']
            new_row['pub_category'] = row['app_category']
            writer_app.writerow(new_row)
        else:
            new_row['pub_id'] = row['site_id']
            new_row['pub_domain'] = row['site_domain']
            new_row['pub_category'] = row['site_category']
            writer_site.writerow(new_row)


