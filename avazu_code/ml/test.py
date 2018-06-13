# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 11:11:29 2018

@author: huang
"""

# coding: utf-8

# # 数据探索
# 

# In[1]:


import pandas as pd
import numpy as np
import sys
sys.path.append('D:\GitHub\jhye_tool\ml')
sys.path.append('D:\GitHub\jhye_tool')



# In[2]:


path='../../kaggle-avazu/avazu_data/'


# In[3]:


"""
• id: ad identifier （广告 ID）
• click: 0/1 for non-click/click （是否被点击，其中 0 为不被点击，1 为不被点击，此列为目
标变量）
• hour: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC. （时间）
• C1 -- anonymized categorical variable （类别型变量）
• banner_pos （广告位置）
• site_id （站点 ID）
• site_domain （站点领域）
• site_category （站点类别）
• app_id （APP ID）
• app_domai
• C14-C21 -- anonymized categorical variables （类别型变量）
"""

#读取数据
train = pd.read_csv(path+"train_01.csv")
train.head()


# In[4]:


#train.info()


# In[5]:


#train.describe()


# # 1、处理hour特征，格式为YYMMDDHH
# 数据为21-31，共11天的数据

# In[6]:


train['day']=np.round(train.hour % 10000 / 100)
train['hour1'] = np.round(train.hour % 100)
train['day_hour'] = (train.day.values - 21) * 24 + train.hour1.values
train['day_hour_prev'] = train['day_hour'] - 1
train['day_hour_next'] = train['day_hour'] + 1


# #  app_id

# In[7]:


train['app_or_web'] = 0
#如果app_id='ecad2386',mobile_dev=1
train.ix[train.app_id.values=='ecad2386', 'app_or_web'] = 1


# In[8]:


#串联app_id和site_id
train['app_site_id'] = np.add(train.app_id.values, train.site_id.values)


# In[9]:


train['dev_id_ip'] = pd.Series(np.add(train.device_id.values , train.device_ip.values)).astype('category').values.codes


# In[10]:


#train.head()


# In[11]:



# In[12]:


filter_t1 = np.logical_and(train.day.values != 20, train.day.values < 31)


# In[16]:


from ml_utils import *


# In[17]:


calcTVTransform(train,'app_id','click',10,filter_t1)


# In[15]:



