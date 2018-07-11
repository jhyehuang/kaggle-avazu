#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 23:12:53 2018

@author: zhijiehuang
"""

import sys
import os
import pandas as pd
cmd="grep ':cnt:' nohup.out"

ret=os.popen(cmd)
ret=ret.readlines()

col_list=[]
col_dict={}
for x in ret:
#    print(x)
    ls=x.split('-')[5].split(':')
    if ls[2].replace('\n','')>'1':
#        print(ls[0],ls[2])
        col_list.append(ls[0].replace(' ',''))
        if int(ls[2].replace('\n',''))>100000:
            col_dict[ls[0].replace(' ','')]=ls[2].replace('\n','')
print(col_list)
print(col_dict.keys())

feature_score=pd.read_csv('/data/feature_score.csv')
feature_score=feature_score[feature_score['importance']>100]
print(list(feature_score['feature'].values))


