
# coding: utf-8

# # 在Minist数据集上进行PCA分析
# 图像数据维数高，而且特征之间（像素之间）相关性很高，因此我们预计用很少的维数就能保留足够多的信息

# In[1]:


#导入必要的工具包
import pandas as pd
import numpy as np
import sys
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
sys.path.append('..')
import time
from joblib import dump, load, Parallel, delayed
import utils
from ml_utils import *
from data_preprocessing import *
from xgboost import XGBClassifier
from sklearn.metrics import log_loss



#sys.path.append(utils.xgb_path)
import xgboost as xgb


import logging


from flags import FLAGS, unparsed

# In[2]:
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)


#读取训练数据和测试数据
train = get_PCA_train_data()
#test = get_PCA_test_data()

y_train = train.click.values
X_train = train.drop("click",axis=1).values
#X_test = test.values 



# 原始输入的特征维数和样本数目
logging.debug('the shape of train_image: {}'.format(X_train.shape))
#logging.debug('the shape of test_image: {}'.format(X_test.shape))


# In[5]:


# 将训练集合拆分成训练集和校验集，在校验集上找到最佳的模型超参数（PCA的维数）
X_train_part, X_val, y_train_part, y_val = train_test_split(X_train,y_train, train_size = 0.8,random_state = 0)


# In[6]:


#拆分后的训练集和校验集的样本数目
logging.debug(X_train_part.shape)
logging.debug(X_val.shape)


# In[7]:

gpu_dict={'tree_method':'gpu_hist',}
# 一个参数点（PCA维数为n）的模型训练和测试，得到该参数下模型在校验集上的预测性能
def n_component_analysis(n, X_train, y_train, X_val, y_val):
    start = time.time()
    
    pca = PCA(n_components=n)
    logging.debug("PCA begin with n_components: {}".format(n));
    pca.fit(X_train)
    
    # 在训练集和测试集降维 
    X_train_pca = pca.transform(X_train)
    
    # 利用SVC训练
    logging.debug('xgb begin')
    alg = XGBClassifier(learning_rate =0.1,
    n_estimators=666,
    max_depth=6,
    min_child_weight=1,
#        gamma=0.1,
#        subsample=0.8,
#        colsample_bytree=0.8,
    scoring='roc_auc',
    objective='binary:logistic',
    eval_metric=['logloss','auc'],
    nthread=-1,
    verbose=2,
#        scale_pos_weight=1,
#        reg_alpha=1.5,
#        reg_lambda=0.5,
    seed=27,
    silent=0,**gpu_dict)
    #Fit the algorithm on the data
    alg.fit(X_train, y_train, eval_metric='logloss')
        
    #Predict training set:
    
    train_predprob = alg.predict_proba(X_val)
    try:
        logloss = log_loss(y_val, train_predprob)
        logging.debug(logloss)
    except:
        pass
    
    try:
        _,lloss = logloss(train_predprob[:,1],y_val)

       #logging.debug model report:
        logging.debug ("logloss of train :" )
        logging.debug(lloss)
    except:
        pass
#    xgb1 = load(FLAGS.tmp_data_path+'xgboost.cv_fin.model.joblib_dat')
#    dtrain_predprob = xgb1.predict_proba(X_train_pca)[:,1]
    
    # 返回accuracy
    accuracy = alg.score(train_predprob[:,1], y_val)
    
    end = time.time()
    logging.debug("accuracy: {}, time elaps:{}".format(accuracy, int(end-start)))
    return accuracy


# In[8]:


# 设置超参数（PCA维数）搜索范围
n_s = np.linspace(0.70, 0.85, num=15)
accuracy = []
for n in n_s:
    tmp = n_component_analysis(n, X_train_part, y_train_part,X_val, y_val)
    accuracy.append(tmp)


# In[9]:


# 绘制不同PCA维数下模型的性能，找到最佳模型／参数（分数最高）
#import matplotlib.pyplot as plt
##get_ipython().magic('matplotlib inline')
#plt.plot(n_s, np.array(accuracy), 'b-')


# In[10]:


#最佳模型参数
#pca = PCA(n_components=0.75)

#根据最佳参数，在全体训练数据上重新训练模型
#pca.fit(X_train)


# In[11]:


#logging.debug(pca.n_components_)


# In[12]:


#logging.debug(pca.explained_variance_ratio_)


# In[13]:


#根据最佳参数，对全体训练数据降维
#X_train_pca = pca.transform(X_train)

#根据最佳参数，对测试数据降维
#X_test_pca = pca.transform(X_test)


# In[14]:


# 降维后的特征维数
#logging.debug(X_train_pca.shape)
#logging.debug(X_test_pca.shape)


# In[15]:


#在降维后的训练数据集上训练SVM分类器
#clf = svm.SVC()
#clf.fit(X_train_pca, y_train)


# In[16]:


# 用在降维后的全体训练数据集上训练的模型对测试集进行测试
#y_predict = clf.predict(X_test_pca)


# In[17]:


#生成提交测试结果
#import pandas as pd
#df = pd.DataFrame(y_predict)
#df.columns=['Label']
#df.index+=1
#df.index.name = 'Imageid'
#df.to_csv('SVC_Minist_submission.csv', header=True)

