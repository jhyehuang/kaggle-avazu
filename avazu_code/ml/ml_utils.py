import numpy as np
import pandas as pd
from sklearn.utils import check_random_state 
import  csv

import time
import sys
import os
filepath=os.path.abspath(__file__)
path=os.path.split(filepath)[0]
sys.path.append(path)
sys.path.append(path+'/..')
print(sys.path)
from joblib import dump, load
import copy
import logging
from flags import FLAGS, unparsed

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)


def print_help():
    logging.debug ("usage: python utils -set_params [tvh=Y|N], [sample_pct]")
    logging.debug ("for example: python utils -set_params N 0.05")

def main():
    if sys.argv[1] == '-set_params' and len(sys.argv) == 4:
        try:
            tvh = sys.argv[2]  # Y/N
            sample_pct = float(sys.argv[3])  #1/0.05
            dump({'pct': sample_pct, 'tvh':tvh}, FLAGS.tmp_data_path + '_params.joblib_dat')
        except:
            logging.debug_help()
    else:
        logging.debug_help()

if __name__ == "__main__":
    main()

def get_agg(group_by, value, func):
    g1 = pd.Series(value).groupby(group_by)
    agg1  = g1.aggregate(func)
    #logging.debug agg1
    r1 = agg1[group_by].values
    return r1

# vn：特征，vn_y：标签列，cred_k：先验强度，r_k=0，
#调用方式：p1 = calcLeaveOneOut2(df1, vn, 'click', n_ks[vn], 0, 0.25, mean0=pred_prev)
def calcLeaveOneOut2(df, vn, vn_y, cred_k, r_k, power, mean0=None, add_count=False):

    #每个特征取值对应的样本组成group
    _key_codes = df[vn].values
    grp1 = df[vn_y].groupby(_key_codes)

    #先验也按每个特征取值进行group
    grp_mean = pd.Series(mean0).groupby(_key_codes)

    #计算每个group中样本的先验均值、样本值之和、样本数目
    mean1 = grp_mean.aggregate(np.mean)
    sum1 = grp1.aggregate(np.sum)
    cnt1 = grp1.aggregate(np.size)
    
    logging.debug(mean1.shape)
    logging.debug(sum1.shape)
    logging.debug(cnt1.shape)

    _sum = sum1[_key_codes].values
    _cnt = cnt1[_key_codes].values
    _mean = mean1[_key_codes].values


    _mean[np.isnan(_mean)] = mean0.mean()
    _cnt[np.isnan(_cnt)] = 0    
    _sum[np.isnan(_sum)] = 0
    logging.debug(_mean[:10])
    logging.debug(_cnt[:10])
    logging.debug(_sum[:10])

    #转化为负值
    _sum -= df[vn_y].values
    _cnt -= 1
    #logging.debug _cnt[:10]

    vn_yexp = 'exp2_'+vn
    #    df[vn_yexp] = (_sum + cred_k * mean0)/(_cnt + cred_k)
    #(总和+随机算子*均值)/(总数+随机算子)/均值    然后开4次方
    diff = np.power((_sum + cred_k * _mean)/(_cnt + cred_k) / _mean, power)

    if vn_yexp in df.columns.values.tolist():
        df[vn_yexp] *= diff
    else:
        df[vn_yexp] = diff 

    if add_count:
        vn_cnt=vn+'_cnt'
        df[vn_cnt] = _cnt

    return diff


def my_lift(order_by, p, y, w, n_rank, dual_axis=False, random_state=0, dither=1e-5, fig_size=None):
    gen = check_random_state(random_state)
    if w is None:
        w = np.ones(order_by.shape[0])
    if p is None:
        p = order_by
    ord_idx = np.argsort(order_by + dither*np.random.uniform(-1.0, 1.0, order_by.size))
    p2 = p[ord_idx]
    y2 = y[ord_idx]
    w2 = w[ord_idx]

    cumm_w = np.cumsum(w2)
    total_w = cumm_w[-1]
    r1 = np.minimum(n_rank, np.maximum(1, 
                    np.round(cumm_w * n_rank / total_w + .4999999)))
    
    df1 = pd.DataFrame({'r': r1, 'pw': p2 * w2, 'yw': y2 * w2, 'w': w2})
    grp1 = df1.groupby('r')
    
    sum_w = grp1['w'].aggregate(np.sum)
    avg_p = grp1['pw'].aggregate(np.sum) / sum_w 
    avg_y = grp1['yw'].aggregate(np.sum) / sum_w
    
    xs = range(1, n_rank+1)
    
    fig, ax1 = plt.subplots()
    if fig_size is None:
        fig.set_size_inches(20, 15)
    else:
        fig.set_size_inches(fig_size)
    ax1.plot(xs, avg_p, 'b--')
    if dual_axis:
        ax2 = ax1.twinx()
        ax2.plot(xs, avg_y, 'r')
    else:
        ax1.plot(xs, avg_y, 'r')
    
    #logging.debug "logloss: ", logloss(p, y, w)
    
    return gini_norm(order_by, y, w)

def logloss(_y, y, weight=None):
    labels=y
#    _y=pred[:,1]
    if weight is None:
        weight = np.ones(labels.size)
    
    _y = np.maximum(1e-7, np.minimum(1 - 1e-7, _y))
    return 'logloss',- np.sum(weight * (labels * np.log(_y) + (1 - labels) * np.log(1 - _y))) / np.sum(weight)

def gini_norm(pred, y, weight=None):

    #equal weight by default
    if weight == None:
        weight = np.ones(y.size)

    #sort actual by prediction
    ord = np.argsort(pred)
    y2 = y[ord]
    w2 = weight[ord]
    #gini by pred
    cumm_y = np.cumsum(y2)
    total_y = cumm_y[-1]
    total_w = np.sum(w2)
    g1 = 1 - 2 * sum(cumm_y * w2) / (total_y * total_w)

    #sort actual by actual
    ord = np.argsort(y)
    y2 = y[ord]
    w2 = weight[ord]
    #gini by actual
    cumm_y = np.cumsum(y2)
    g0 = 1 - 2 * sum(cumm_y * w2) / (total_y * total_w)

    return g1/g0

#如果俩个样本集内重叠就是按照 1赋值，其他不重回情况取1的均值
def mergeLeaveOneOut2(df, dfv, vn):
    _key_codes = df[vn].values.codes
    vn_yexp = 'exp2_'+vn

    grp1 = df[vn_yexp].groupby(_key_codes)
    _mean1 = grp1.aggregate(np.mean)
    
    _mean = _mean1[dfv[vn].values.codes].values
    
    _mean[np.isnan(_mean)] = _mean1.mean()

    return _mean
    
#vn: 特征列，vn_y：click列
# cred_k：先验强度
def calcTVTransform(df, vn, vn_y, cred_k, filter_train, mean0=None):
    #先验
#    logging.debug(df)
    if mean0 is None:
        mean0 = df.ix[filter_train, vn_y].mean()
        logging.debug(str(mean0))
    else:
        mean0 = mean0[~filter_train]

    #似然
    logging.debug(filter_train)
    df['_key1'] = df[vn].astype('category').values.codes
    df_yt = df.ix[filter_train, ['_key1', vn_y]]
    #df_y.set_index([')key1'])
    logging.debug(df_yt)

    #按照参数输入的key对内容进行分组
    grp1 = df_yt.groupby(['_key1'])
    #通过 key 分组获得 真值对应的 sum值
    sum1 = grp1[vn_y].aggregate(np.sum)
    #通过 key 分组获得 真值对应的 元素个数    在取值为0时，和sum不同
    cnt1 = grp1[vn_y].aggregate(np.size)
    
    logging.debug(sum1)
    logging.debug(cnt1)
#    vn_sum = 'sum_' + vn
#    vn_cnt = 'cnt_' + vn
    
    # 加工指定域以外的 值
#    logging.debug((~filter_train).shape)
    v_codes = df.ix[filter_train, '_key1']
    logging.debug(v_codes.shape)
    # 按照
    _sum = sum1[v_codes].values
    _cnt = cnt1[v_codes].values
    logging.debug(_sum.shape)
    logging.debug(_cnt.shape)

    _cnt[np.isnan(_cnt)] = 0
    _sum[np.isnan(_sum)] = 0

    # 后验均值
    r = {}
    r['exp'] = (_sum + cred_k * mean0)/(_cnt + cred_k)
    r['cnt'] = _cnt
    logging.debug(r)
    return r

def cntDualKey(df, vn, vn2, key_src, key_tgt, fill_na=False):
    
    logging.debug ("build src key")
    _key_src = np.add(df[key_src].astype('string').values, df[vn].astype('string').values)
    logging.debug ("build tgt key")
    _key_tgt = np.add(df[key_tgt].astype('string').values, df[vn].astype('string').values)
    
    if vn2 is not None:
        _key_src = np.add(_key_src, df[vn2].astype('string').values)
        _key_tgt = np.add(_key_tgt, df[vn2].astype('string').values)

    logging.debug ("aggreate by src key")
    grp1 = df.groupby(_key_src)
    # 获得vn 在分组字段上的个数
    cnt1 = grp1[vn].aggregate(np.size)
    
    logging.debug ("map to tgt key")
#    vn_sum = 'sum_' + vn + '_' + key_src + '_' + key_tgt
    #  _key_src  count
    _cnt = cnt1[_key_tgt].values

    if fill_na is not None:
        logging.debug ("fill in na")
        _cnt[np.isnan(_cnt)] = fill_na    

    vn_cnt_tgt = 'cnt_' + vn + '_' + key_tgt
    if vn2 is not None:
        vn_cnt_tgt += '_' + vn2
    df[vn_cnt_tgt] = _cnt

def my_grp_cnt(group_by, count_by):
    _ts = time.time()
    #按照group_by 排序 ，返回索引
    _ord = np.lexsort((count_by, group_by))
    logging.debug (time.time() - _ts)
    _ts = time.time()    
#    _ones = pd.Series(np.ones(group_by.size))
    
#    logging.debug (time.time() - _ts)
#    _ts = time.time()    
    #_cs1 = _ones.groupby(group_by[_ord]).cumsum().values
    # group_by 的尺寸
    _cs1 = np.zeros(group_by.size)
    # 初始化
    _prev_grp = '___'
    
    same_cnt = 0
    for i in range(1, group_by.size):
        # 读取到索引
        i0 = _ord[i]
        #判断前一个域值是否和当前相等
        if _prev_grp == group_by[i0]:
            #判断count_by 的前一个是否等于当前
            # true  running +1
            if count_by[_ord[i-1]] != count_by[i0]: 
                same_cnt += 1
        else:
            same_cnt = 1
            _prev_grp = group_by[i0]
        if i == group_by.size - 1 or group_by[i0] != group_by[_ord[i+1]]:
            #如果   i  max 或者 group_by 当前不等于下一个 
            j = i
            while True:
                j0 = _ord[j]
                _cs1[j0] = same_cnt
                if j == 0 or group_by[_ord[j-1]] != group_by[j0]:
                    break
                j -= 1
        #排序之后，找出相等的值得数量
            
    logging.debug (time.time() - _ts)
    if True:
        return _cs1
    else:
        _ts = time.time()    

        org_idx = np.zeros(group_by.size, dtype=np.int)
        logging.debug (time.time() - _ts)
        _ts = time.time()    
        org_idx[_ord] = np.asarray(range(group_by.size))
        logging.debug (time.time() - _ts)
        _ts = time.time()    

        return _cs1[org_idx]
    
def my_cnt(group_by):
    _ts = time.time()
    _ord = np.argsort(group_by)
    logging.debug (time.time() - _ts)
    _ts = time.time()    
    #_cs1 = _ones.groupby(group_by[_ord]).cumsum().values
    _cs1 = np.zeros(group_by.size)
    _prev_grp = '___'
    runnting_cnt = 0
    for i in range(1, group_by.size):
        i0 = _ord[i]
        if _prev_grp == group_by[i0]:
            running_cnt += 1
        else:
            running_cnt = 1
            _prev_grp = group_by[i0]
        if i == group_by.size - 1 or group_by[i0] != group_by[_ord[i+1]]:
            j = i
            while True:
                j0 = _ord[j]
                _cs1[j0] = running_cnt
                if j == 0 or group_by[_ord[j-1]] != group_by[j0]:
                    break
                j -= 1
            
    logging.debug (time.time() - _ts)
    return _cs1

def my_grp_value_diff(group_by, order_by, value):
    _ts = time.time()
    _ord = np.lexsort((order_by, group_by))
    logging.debug (time.time() - _ts)
    _ts = time.time()    
    _ones = pd.Series(np.ones(group_by.size))
    logging.debug (time.time() - _ts)
    _ts = time.time()    
    #_cs1 = _ones.groupby(group_by[_ord]).cumsum().values
    _cs1 = np.zeros(group_by.size)
    _prev_grp = '___'
    for i in range(1, group_by.size):
        i0 = _ord[i]
        if _prev_grp == group_by[i0]:
            _cs1[i0] = value[_ord[i]] - value[_ord[i-1]]
        else:
            _cs1[i0] = 1e7
            _prev_grp = group_by[i0]
    logging.debug (time.time() - _ts)
    
    return np.minimum(_cs1, 1e7)

def my_grp_idx(group_by, order_by):
    _ts = time.time()
    _ord = np.lexsort((order_by, group_by))
    logging.debug (time.time() - _ts)
    _ts = time.time()    
#    _ones = pd.Series(np.ones(group_by.size))
    logging.debug (time.time() - _ts)
    _ts = time.time()    
    #_cs1 = _ones.groupby(group_by[_ord]).cumsum().values
    _cs1 = np.zeros(group_by.size)
    _prev_grp = '___'
    for i in range(1, group_by.size):
        i0 = _ord[i]
        if _prev_grp == group_by[i0]:
            _cs1[i] = _cs1[i - 1] + 1
        else:
            _cs1[i] = 1
            _prev_grp = group_by[i0]
    logging.debug (time.time() - _ts)
    _ts = time.time()    
    
    org_idx = np.zeros(group_by.size, dtype=np.int)
    logging.debug (time.time() - _ts)
    _ts = time.time()    
    # 排序之后的 有序列表对应的下标
    org_idx[_ord] = np.asarray(range(group_by.size))
    logging.debug (time.time() - _ts)
    _ts = time.time()    

    return _cs1[org_idx]

def calcDualKey(df, vn, vn2, key_src, key_tgt, vn_y, cred_k, mean0=None, add_count=False, fill_na=False):
    if mean0 is None:
        mean0 = df[vn_y].mean()
    
    logging.debug ("build src key")
    _key_src = np.add(df[key_src].astype('string').values, df[vn].astype('string').values)
    logging.debug ("build tgt key")
    _key_tgt = np.add(df[key_tgt].astype('string').values, df[vn].astype('string').values)
    
    if vn2 is not None:
        _key_src = np.add(_key_src, df[vn2].astype('string').values)
        _key_tgt = np.add(_key_tgt, df[vn2].astype('string').values)

    logging.debug ("aggreate by src key")
    grp1 = df.groupby(_key_src)
    sum1 = grp1[vn_y].aggregate(np.sum)
    cnt1 = grp1[vn_y].aggregate(np.size)
    
    logging.debug ("map to tgt key")
#    vn_sum = 'sum_' + vn + '_' + key_src + '_' + key_tgt
    _sum = sum1[_key_tgt].values
    _cnt = cnt1[_key_tgt].values

    if fill_na:
        logging.debug ("fill in na")
        _cnt[np.isnan(_sum)] = 0    
        _sum[np.isnan(_sum)] = 0

    logging.debug ("calc exp")
    if vn2 is not None:
        vn_yexp = 'exp_' + vn + '_' + vn2 + '_' + key_src + '_' + key_tgt
    else:
        vn_yexp = 'exp_' + vn + '_' + key_src + '_' + key_tgt
    df[vn_yexp] = (_sum + cred_k * mean0)/(_cnt + cred_k)

    if add_count:
        logging.debug ("add counts")
        vn_cnt_src = 'cnt_' + vn + '_' + key_src
        df[vn_cnt_src] = _cnt
        grp2 = df.groupby(_key_tgt)
        cnt2 = grp2[vn_y].aggregate(np.size)
        _cnt2 = cnt2[_key_tgt].values
        vn_cnt_tgt = 'cnt_' + vn + '_' + key_tgt
        df[vn_cnt_tgt] = _cnt2

def get_set_diff(df, vn, f1, f2):
    #logging.debug(df[vn].values.sum())
    set1 = set(np.unique(df[vn].values[f1]))
    set2 = set(np.unique(df[vn].values[f2]))
    set2_1 = set2 - set1
    logging.debug (vn, '\t', len(set1), '\t', len(set2), '\t', len(set2_1))
    return len(set2_1) * 1.0 / len(set2)


def calc_exptv(data1,data2,data3,vn_list,add_count=False):
    day_exps = {}
    cred_k=10
    day=pd.read_csv(data2)
    train=pd.read_csv(data1)
    num=pd.read_csv(data3)
#    train=train.join(day, on='key')
#    del day

    new_list=[]
    one_day=day[['one_day']]
    one_day.one_day=one_day.one_day.map(int).map(str)
    days_list=list(set(one_day.one_day.values))


    for day_v in days_list:
        day_exps[day_v]={}
#    vn_list=train.columns.values.tolist()
    t3= pd.DataFrame(columns=['one_day','click'])  
    t3['one_day']=day['one_day'].values
    t3['click']=num['click'].values
#    t3['id']=num['id'].values
    one_drop_list=[]
    for one in vn_list:
        t1=train[[one]]
        one_drop_list.append(one)
#        t3[one] = t1[one].values
        two_vn_list=list(set(vn_list).difference(set(one_drop_list)))
        if len(two_vn_list)<1:
            break
        for two in two_vn_list:
            t2=train[[two]]
            vn=one+two
            logging.debug(vn)
            logging.debug(t1.shape)
            logging.debug(t2.shape)
            logging.debug(t3.shape)

            t3[vn] = pd.Series(np.add(t1[one].astype('str').values , t2[two].astype('str').values)).astype('category').values.codes
            for day_v in days_list:
                print (day_v)
                filter_t1 = (one_day.one_day.values ==day_v )
                logging.debug(filter_t1.shape)
                
                day_exps[day_v][vn] = calcTVTransform(t3, vn, 'click', cred_k, filter_t1)
            new_list.append(vn)
    t3.drop(['one_day'], axis=1,inplace = True)
    t3.drop(['click'], axis=1,inplace = True)
    t3.to_csv(FLAGS.tmp_data_path+'two_col_join.csv',index=False)
    ret=dump(day_exps, FLAGS.tmp_data_path+'day_exps.joblib_dat') 
    ret=dump(new_list, FLAGS.tmp_data_path+'new_list.joblib_dat') 
    ret=dump(days_list, FLAGS.tmp_data_path+'days_list.joblib_dat') 
    ret=dump(one_day, FLAGS.tmp_data_path+'one_day.joblib_dat')
    logging.debug(ret)
    del t3,day_exps
    
def calc_exptv_cnt():
    t3=pd.read_csv(FLAGS.tmp_data_path+'two_col_join.csv')
    drop_col_list=t3.columns
#    drop_col_list.remove('id')
    day_exps=load(FLAGS.tmp_data_path+'day_exps.joblib_dat')
    new_list=load(FLAGS.tmp_data_path+'new_list.joblib_dat')
    days_list=load(FLAGS.tmp_data_path+'days_list.joblib_dat')
    one_day=load(FLAGS.tmp_data_path+'one_day.joblib_dat')
    t4= t3   
    two_new_list=[]
    for vn_key in new_list:
        vn_key_data=t3[vn_key]
        vn_exp = 'exptv_'+vn_key
        vn_cnt = 'cnttv_'+vn_key
        logging.debug(vn_key)
        t4[vn_exp] = np.zeros(vn_key_data.shape[0])
        t4[vn_cnt] = np.zeros(vn_key_data.shape[0])
        two_new_list.append(vn_exp)
        two_new_list.append(vn_cnt)
        for day_v in days_list:
            m=(one_day.one_day.values == day_v)
            logging.debug(one_day.one_day.values.shape)
            logging.debug(m)
            t4.loc[m, vn_exp]=day_exps[day_v][vn_key]['exp']
            t4.loc[m, vn_cnt]=day_exps[day_v][vn_key]['cnt']
    t4.drop(drop_col_list, axis=1,inplace = True)
    t4.to_csv(FLAGS.tmp_data_path+'two_col_join_cnt.csv',index=False)
    del t3,t4
    return two_new_list

def col_anly(data,col_name):
    #读取数据
    train = data
    logging.debug(train.head())
    logging.debug(col_name)
    a=train[col_name].value_counts()
    return a

def check_col_count_less_11(data,col_name):
    a=col_anly(data,col_name)
    return dict(a[a.values<10])

def col_plt(data,col_name,plt):
    #读取数据
    plt.scatter(range(data.shape[0]), data.values,color='purple')
    plt.title("count of "+col_name);
    
def col_countplot(col_name,plt,sns):
    #读取数据
    train_one = pd.read_csv(FLAGS.tmp_data_path+col_name)
    sns.countplot(x=col_name, data=train_one)
    plt.xlabel(col_name)
    plt.ylabel('Number of '+col_name);
    plt.show()
    
def two_col_plt(col_name,plt,sns):
    train = pd.read_csv(FLAGS.tmp_data_path+col_name)
    print (train.head())
    flights = train.pivot('device_conn_type',)
    
    plt.subplots()
    sns.heatmap(flights,annot=True)
    
    # Mask unimportant features
    #sns.heatmap(correlationMatrix, mask=correlationMatrix < 1, cbar=False)
    plt.show()
    

    
def sns_factorplot(train_one,col_name,plt,sns,y=None,hue=None,row=None,col=None,kind="box"):
#    train_one = pd.read_csv(FLAGS.tmp_data_path+col_name)
    sns.factorplot(x=col_name, y=y, 
                   hue=y, data=train_one,
                   row=row, col=col, 
                   col_wrap=None, estimator=np.mean, ci=95, 
                   n_boot=1000, units=None, order=None, 
                   hue_order=None, row_order=None, 
                   col_order=None, kind=kind, size=4, 
                   aspect=1, orient=None, color=None, 
                   palette=None, legend=True, 
                   legend_out=True, sharex=True, 
                   sharey=True, margin_titles=False, 
                   facet_kws=None)
    plt.show()
    

def procdess_col(train_one,col_name):
    _count = train_one[col_name].value_counts()
    logging.debug(_count)
    
    def cnt_xx(xx):
       return  _count.index.values[_count.values >= np.percentile(_count.values, xx)] 

    cnt_=cnt_xx(90)
    logging.debug(cnt_)
    def func_xx(x):
        return  1 if x in cnt_ else 0
    train_one['top_10_'+col_name] = train_one[col_name].apply(func_xx)
    logging.debug('top_10_')
    cnt_=cnt_xx(75)
    logging.debug(cnt_)
    def func_xx(x):
        return  1 if x in cnt_ else 0
    train_one['top_25_'+col_name] = train_one[col_name].apply(func_xx)
    logging.debug('top_25_')
    cnt_=cnt_xx(95)
    logging.debug(cnt_)
    def func_xx(x):
        return  1 if x in cnt_ else 0
    train_one['top_5_'+col_name] = train_one[col_name].apply(func_xx)
    logging.debug('top_5_')
    cnt_=cnt_xx(50)
    logging.debug(cnt_)
    def func_xx(x):
        return  1 if x in cnt_ else 0
    train_one['top_50_'+col_name] = train_one[col_name].apply(func_xx)
    logging.debug('top_50_')
    cnt_=cnt_xx(99)
    logging.debug(cnt_)
    def func_xx(x):
        return  1 if x in cnt_ else 0
    train_one['top_1_'+col_name] = train_one[col_name].apply(func_xx)
    logging.debug('top_1_')
    cnt_=cnt_xx(98)
    logging.debug(cnt_)
    def func_xx(x):
        return  1 if x in cnt_ else 0
    train_one['top_2_'+col_name] = train_one[col_name].apply(func_xx)
    logging.debug('top_2_')
    cnt_=cnt_xx(85)
    logging.debug(cnt_)
    def func_xx(x):
        return  1 if x in cnt_ else 0
    train_one['top_15_'+col_name] = train_one[col_name].apply(func_xx)
    logging.debug('top_15_')
    cnt_=cnt_xx(80)
    logging.debug(cnt_)
    def func_xx(x):
        return  1 if x in cnt_ else 0
    train_one['top_20_'+col_name] = train_one[col_name].apply(func_xx)
    logging.debug('top_20_')
    cnt_=cnt_xx(30)
    logging.debug(cnt_)
    def func_xx(x):
        return  1 if x in cnt_ else 0
    train_one['top_30_'+col_name] = train_one[col_name].apply(func_xx)
    logging.debug('top_30_')
    
    logging.debug(_count)
    return train_one