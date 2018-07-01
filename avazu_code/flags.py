#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import datetime

import pytz


#pad='win'
pad='tiny'
#pad='linux'
# pool.supportxmr.com:3333
#bounced zombie fossil wayside enforce visited arsenic educated serving linen mittens argue wayside
# 42MXU6g6N1q46FfEFWFqGF5xu6vBEHh4CUdE9iJGevPUPexfTa3eGgjh46gBGieH4gHwb7QCCCiF7HNNTwPQ3V59HK1rVm6
# MinerIdentifier

FLAGS, unparsed='',''

tz = pytz.timezone('Asia/Shanghai')
current_time = datetime.datetime.now(tz)

gdbt_param = {'max_depth':15, 'eta':.02, 'objective':'binary:logistic', 'verbose':0,
         'subsample':1.0, 'min_child_weight':50, 'gamma':0,
         'nthread': -1, 'colsample_bytree':.5,  'seed': 999,
         'early_stopping_rounds':10,'num_boost_round':1000}

def parse_args(check=True):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--tool_dir', type=str, default='/home/zhijiehuang/github/jhye_tool',
                        help='path to my tool.')
    
    parser.add_argument('--tool_ml_dir', type=str, default='/home/zhijiehuang/github/jhye_tool/ml',
                        help='path to my ml tool.')
    
    parser.add_argument('--output_dir', type=str, default='/data/',
                        help='path to save log and checkpoint.')
    
    parser.add_argument('--train_set_path', type=str, default='/home/zhijiehuang/github/data/',
                        help='path to save train test and .')

    parser.add_argument('--tmp_data_path', type=str, default='/data/',
                        help='path to QuanSongCi.txt')

    parser.add_argument('--train_job_name', type=str, default='trian_job',
                        help='number of time steps of one sample.')

    parser.add_argument('--src_train_path', type=str, default='/home/zhijiehuang/github/data/train.csv',
                        help='src_train_path.')
    
    parser.add_argument('--src_test_path', type=str, default='/home/zhijiehuang/github/data/test.csv',
                        help='src_test_path.')

    parser.add_argument('--dst_app_path', type=str, default='/data/writer_app.csv',
                        help='dst_app_path.')

    parser.add_argument('--dst_site_path', type=str, default='/data/writer_site.csv',
                        help='dst_site_path ')

    parser.add_argument('--sample_pct', type=float, default=1,
                        help='batch size to use.')

    parser.add_argument('--xgb_n_trees', type=int, default=300,
                        help='xgb_n_trees')

    parser.add_argument('--reverse_dictionary', type=str, default='reverse_dictionary.json',
                        help='path to reverse_dictionary.json.')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    
    parser.add_argument('--rf_feature_list', type=list, default=['C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'banner_pos', 'device_type', 'device_conn_type'],
                        help='rf_feature_list ')
    
    parser.add_argument('--n_trees', type=int, default=300,
                        help='n_trees ')
    
    parser.add_argument('--gbdt_param', type=dict, default=gdbt_param,
                        help='gdbt_param ')
    
    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS, unparsed

def win_parse_args(check=True):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--tool_dir', type=str, default='d:/GitHub/jhye_tool',
                        help='path to my tool.')
    
    parser.add_argument('--tool_ml_dir', type=str, default='d:/GitHub/jhye_tool/ml',
                        help='path to my ml tool.')
    
    parser.add_argument('--output_dir', type=str, default='f:/GitHub/data/avazu_data/output',
                        help='path to save log and checkpoint.')
    
    parser.add_argument('--train_set_path', type=str, default='d:/GitHub/kaggle-avazu/avazu_data',
                        help='path to save train test and .')

    parser.add_argument('--tmp_data_path', type=str, default='f:/GitHub/data/avazu_data/output/',
                        help='tmp_data_path')

    parser.add_argument('--train_job_name', type=str, default='trian_job',
                        help='number of time steps of one sample.')

    parser.add_argument('--src_train_path', type=str, default='d:/GitHub/data/avazu_data/train_01.csv',
                        help='src_train_path.')
    
    parser.add_argument('--src_test_path', type=str, default='d:/GitHub/data/avazu_data/test_01.csv',
                        help='src_test_path.')

    parser.add_argument('--dst_app_path', type=str, default='f:/GitHub/data/avazu_data/output/writer_app.csv',
                        help='dst_app_path.')

    parser.add_argument('--dst_site_path', type=str, default='f:/GitHub/data/avazu_data/output/writer_site.csv',
                        help='dst_site_path ')

    parser.add_argument('--sample_pct', type=float, default=1,
                        help='batch size to use.')

    parser.add_argument('--xgb_n_trees', type=int, default=300,
                        help='xgb_n_trees')

    parser.add_argument('--reverse_dictionary', type=str, default='reverse_dictionary.json',
                        help='path to reverse_dictionary.json.')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    
    parser.add_argument('--rf_feature_list', type=list, default=['C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'banner_pos', 'device_type', 'device_conn_type'],
                        help='rf_feature_list ')
    
    parser.add_argument('--n_trees', type=int, default=300,
                        help='n_trees ')
    
    parser.add_argument('--gbdt_param', type=dict, default=gdbt_param,
                        help='gdbt_param ')
    
    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS, unparsed


def tiny_parse_args(check=True):
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('--tool_dir', type=str, default='d:/GitHub/jhye_tool',
                        help='path to my tool.')
    
    parser.add_argument('--tool_ml_dir', type=str, default='d:/GitHub/jhye_tool/ml',
                        help='path to my ml tool.')
    
    parser.add_argument('--output_dir', type=str, default='f:/GitHub/data/avazu_data/output',
                        help='path to save log and checkpoint.')
    
    parser.add_argument('--train_set_path', type=str, default='d:/GitHub/kaggle-avazu/avazu_data',
                        help='path to save train test and .')

    parser.add_argument('--tmp_data_path', type=str, default='f:/GitHub/data/avazu_data/output/',
                        help='tmp_data_path')

    parser.add_argument('--train_job_name', type=str, default='trian_job',
                        help='number of time steps of one sample.')

    parser.add_argument('--tmp_data_path', type=str, default='/data/jhyehuang/crt1537/',
                        help='path to QuanSongCi.txt')
    
    parser.add_argument('--test_data_path', type=str, default='/data/jhyehuang/crttest/',
                        help='path to QuanSongCi.txt')
    
    parser.add_argument('--out_data_path', type=str, default='/output/',
                        help='path to out')

    
    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS, unparsed

if pad=='win':
    FLAGS, unparsed = win_parse_args()
elif pad=='linux':
    FLAGS, unparsed = parse_args()
elif pad=='tiny':
    FLAGS, unparsed = tiny_parse_args()
#for x in dir(FLAGS):
#    print(getattr(FLAGS, x))
