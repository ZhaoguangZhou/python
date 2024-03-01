# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 20:28:45 2024

@author: 86158
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from scipy.stats import zscore

#数据标准化列名表
columns_to_normalize = list()
#文件路径
offline_path="D:\\Zdata\\ccf_offline_stage1_train.csv"
online_path="D:\\Zdata\\ccf_online_stage1_train.csv"
test_path="D:\\Zdata\\ccf_offline_stage1_test_revised.csv"


#数据预处理
def preprocess(df):
    data = df.copy()
    
    # Discount_rate处理
    if "Distance" in data.columns:
        #消费卷类型判断(off满减)
        data["is_off"]=data["Discount_rate"].apply(
            lambda r:1 if ':' in str(r) else 0
            )
        
        #增加min_cost列,对满减类型卷给出消费最低价格，折扣卷赋值0
        data["min_cost"]=data["Discount_rate"].apply(
            lambda r: 0 if ':' not in str(r) else float(str(r).split(':')[0])
            )
        
        #将Discount_rate转换为折扣率
        data["discount_rate"]=data[data["Discount_rate"].notnull()]["Discount_rate"].apply(
            lambda r:float(r) if ':' not in r else 
            (float(str(r).split(':')[0])-float(str(r).split(':')[1]))/float(str(r).split(':')[0])
            )
    
    
    # Distance处理(将空值替换为均值)
    if "Distance" in data.columns:
        distance_mean=data["Distance"].sum()/data[data["Distance"].notnull()].shape[0]
        distance_mean=float(round(distance_mean))
        data["Distance"].fillna(value=distance_mean,inplace=True)
    
    # Date_received和Date处理
    data['Date_received'] = pd.to_datetime(data['Date_received'], format='%Y%m%d')
    if 'Date' in data.columns.tolist():  
        data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
    
    columns_to_normalize.extend(['min_cost'])
    
    return data


#获取正样本集合
def get_data_ps(df):
    data=df.copy()
    
    #获取正样本集合(领卷并消费) positive sample
    data_ps=data[(data["Coupon_id"].notnull() & data["Date"].notnull())]
    
    return data_ps


#获取负样本集合
def get_data_ns(df):
    data=df.copy()
    
    #获取负样本集合(领卷未消费) negative sample
    data_ns=data[(data["Coupon_id"].notnull() & data["Date"].isnull())]
    
    return data_ns


#获取普通样本
def get_data_os(df):
    data=df.copy()
    
    #获取普通样本(未领卷并消费) ordinary sample
    data_os=data[(data["Coupon_id"].isnull() & data["Date"].notnull())]
    
    return data_os


#获取样本集合(正、负样本并集)
def get_data_ss(df):
    data=df.copy()
    
    #获取样本集合(正、负样本并集) sample set
    data_ss=data[data["Coupon_id"].notnull()]
    
    return data_ss


#添加目标值
def add_target(df):
    data = df.copy()
    
    #领券后15天内消费为1,否则为0
    data['target'] = list(map(lambda x, y: 1 if (x - y).total_seconds() / (60 * 60 * 24) <= 15 else 0, data["Date"],
                             data["Date_received"]))
    
    return data


#划分数据集
def divide_dataset(df):
    train_feature=df.loc[( (df['Date']>=datetime(2016,1,1)) & (df['Date']<=datetime(2016,4,13)) )| 
                         ( (df['Date']==np.nan) & (df['Date_received']>=datetime(2016,1,1)) &
                           (df['Date_received']<=datetime(2016,4,13))
                         )]

    
    verify_feature=df.loc[( (df['Date']>=datetime(2016,2,1)) & (df['Date']<=datetime(2016,5,14)) )| 
                          ( (df['Date']==np.nan) & (df['Date_received']>=datetime(2016,2,1)) &
                            (df['Date_received']<=datetime(2016,5,14))
                        )]                                                  
    
    
    test_feature=df.loc[( (df['Date']>=datetime(2016,3,15)) & (df['Date']<=datetime(2016,6,30)) )| 
                        ( (df['Date']==np.nan) & (df['Date_received']>=datetime(2016,3,15)) &
                          (df['Date_received']<=datetime(2016,6,30))
                        )]
    
    train_target=None
    verify_target=None                           
    if 'Distance' in df.columns.tolist():
        train_target=df[(df['Date_received']>=datetime(2016,4,14)) & (df['Date_received']<=datetime(2016,5,14))]
        verify_target=df[(df['Date_received']>=datetime(2016,5,15)) & (df['Date_received']<=datetime(2016,6,15))]

        return train_feature,train_target,verify_feature, verify_target,test_feature

                           
    return train_feature,verify_feature,test_feature


#获取主要特征
def get_main_feature(df):
    data = df.copy()
    
    
    #特征数据集
    merchant_feature=data['Merchant_id'].copy() #商家特征
    coupon_feature=data['Coupon_id'].copy()     #优惠卷特征
    user_feature=data['User_id'].copy()         #用户特征
    
    #商家特征
    
    #各商家 用户购物数
    f=data[data['Date'].notnull()]
    f['buy_num_every_merchant']=1
    f=f[['Merchant_id','buy_num_every_merchant']]
    pivot=pd.DataFrame(
        pd.pivot_table(f,index=['Merchant_id'],
                       values=['buy_num_every_merchant'],
                       aggfunc=len)).reset_index()
    merchant_feature=pd.merge(merchant_feature, pivot,on='Merchant_id',how='left')
    
    #各商家 用户数
    f=data[data['Date'].notnull()]
    f['user_num_every_merchant']=1
    f=f[['Merchant_id','User_id','user_num_every_merchant']]
    f.drop_duplicates(subset=['User_id'], keep='first', inplace=True)
    f=f[['Merchant_id','user_num_every_merchant']]
    pivot=pd.DataFrame(
        pd.pivot_table(f,index=['Merchant_id'],
                       values=['user_num_every_merchant'],
                       aggfunc=len)).reset_index()
    merchant_feature=pd.merge(merchant_feature, pivot,on='Merchant_id',how='left')
    
    
    
    #各商家 用户用卷数
    f=get_data_ps(data)
    f['ps_num_every_merchant']=1
    f=f[['Merchant_id','ps_num_every_merchant']]
    pivot=pd.DataFrame(
        pd.pivot_table(f,index=['Merchant_id'],
                       values=['ps_num_every_merchant'],
                       aggfunc=len)).reset_index()
    merchant_feature=pd.merge(merchant_feature, pivot,on='Merchant_id',how='left')


    #各商家 发卷数
    f=get_data_ss(data)
    f['ss_num_every_merchant']=1
    f=f[['Merchant_id','ss_num_every_merchant']]
    pivot=pd.DataFrame(
        pd.pivot_table(f,index=['Merchant_id'],
                       values=['ss_num_every_merchant'],
                       aggfunc=len)).reset_index()
    merchant_feature=pd.merge(merchant_feature, pivot,on='Merchant_id',how='left')
    
    
    #各商家 用户用卷消费折扣均值
    f=get_data_ps(data)
    f=f[['Merchant_id','discount_rate']].groupby('Merchant_id').agg('mean')
    f['ps_discount_mean_every_merchant']=f['discount_rate']
    f.drop(labels='discount_rate',axis=1,inplace=True)
    f.reset_index()
    merchant_feature=pd.merge(merchant_feature, f,on='Merchant_id',how='left')
    
    #各商家 用户用卷消费折扣最小值
    f=get_data_ps(data)
    f=f[['Merchant_id','discount_rate']].groupby('Merchant_id').agg('min')
    f['ps_discount_min_every_merchant']=f['discount_rate']
    f.drop(labels='discount_rate',axis=1,inplace=True)
    f.reset_index()
    merchant_feature=pd.merge(merchant_feature, f,on='Merchant_id',how='left')
    
    
    #各商家 用户用卷消费折扣最大值
    f=get_data_ps(data)
    f=f[['Merchant_id','discount_rate']].groupby('Merchant_id').agg('max')
    f['ps_discount_max_every_merchant']=f['discount_rate']
    f.drop(labels='discount_rate',axis=1,inplace=True)
    f.reset_index()
    merchant_feature=pd.merge(merchant_feature, f,on='Merchant_id',how='left')
    
    
    #各商家 用户用卷消费距离均值
    f=get_data_ps(data)
    f=f[['Merchant_id','Distance']]
    f=f[['Merchant_id','Distance']].groupby('Merchant_id').agg('mean')
    f['ps_distance_mean_every_merchant']=f['Distance'].apply(lambda x: int( round(x) ) )
    f.drop(labels='Distance',axis=1,inplace=True)
    f.reset_index()
    merchant_feature=pd.merge(merchant_feature, f,on='Merchant_id',how='left')
    
    #各商家 用户用卷消费距离最小值
    f=get_data_ps(data)
    f=f[['Merchant_id','Distance']]
    f=f[['Merchant_id','Distance']].groupby('Merchant_id').agg('min')
    f['ps_distance_min_every_merchant']=f['Distance'].apply(lambda x: int( round(x) ) )
    f.drop(labels='Distance',axis=1,inplace=True)
    f.reset_index()
    merchant_feature=pd.merge(merchant_feature, f,on='Merchant_id',how='left')
    
    #各商家 用户用卷消费距离最大值
    f=get_data_ps(data)
    f=f[['Merchant_id','Distance']]
    f=f[['Merchant_id','Distance']].groupby('Merchant_id').agg('max')
    f['ps_distance_max_every_merchant']=f['Distance'].apply(lambda x: int( round(x) ) )
    f.drop(labels='Distance',axis=1,inplace=True)
    f.reset_index()
    merchant_feature=pd.merge(merchant_feature, f,on='Merchant_id',how='left')
    
    #优惠卷特征
    
    #各优惠卷 使用次数
    f=get_data_ps(data)
    f['used_num_every_coupon']=1
    f=f[['Coupon_id','used_num_every_coupon']]
    pivot=pd.DataFrame(
        pd.pivot_table(f,index=['Coupon_id'],
                       values=['used_num_every_coupon'],
                       aggfunc=len)).reset_index()
    coupon_feature=pd.merge(coupon_feature, pivot,on='Coupon_id',how='left')
    
    #各优惠卷 发卷数
    f=get_data_ss(data)
    f['ss_num_every_coupon']=1
    f=f[['Coupon_id','ss_num_every_coupon']]
    pivot=pd.DataFrame(
        pd.pivot_table(f,index=['Coupon_id'],
                       values=['ss_num_every_coupon'],
                       aggfunc=len)).reset_index()
    coupon_feature=pd.merge(coupon_feature, pivot,on='Coupon_id',how='left')
    
    
    #用户特征
    
    #各用户 消费数
    f=data[data['Date'].notnull()]
    f['buy_num_every_user']=1
    f=f[['User_id','buy_num_every_user']]
    pivot=pd.DataFrame(
        pd.pivot_table(f,index=['User_id'],
                       values=['buy_num_every_user'],
                       aggfunc=len)).reset_index()
    user_feature=pd.merge(user_feature, pivot,on='User_id',how='left')
    
    #各用户 领卷数
    f=get_data_ss(data)
    f['received_num_every_user']=1
    f=f[['User_id','received_num_every_user']]
    pivot=pd.DataFrame(
        pd.pivot_table(f,index=['User_id'],
                       values=['received_num_every_user'],
                       aggfunc=len)).reset_index()
    user_feature=pd.merge(user_feature, pivot,on='User_id',how='left')
    
    #各用户 领卷消费数
    f=get_data_ps(data)
    f['ps_num_every_user']=1
    f=f[['User_id','ps_num_every_user']]
    pivot=pd.DataFrame(
        pd.pivot_table(f,index=['User_id'],
                       values=['ps_num_every_user'],
                       aggfunc=len)).reset_index()
    user_feature=pd.merge(user_feature, pivot,on='User_id',how='left')
    
    
    columns_to_normalize.extend(merchant_feature.columns.tolist())
    columns_to_normalize.extend(coupon_feature.columns.tolist())
    columns_to_normalize.extend(user_feature.columns.tolist())
    columns_to_normalize.remove('Merchant_id')
    columns_to_normalize.remove('Coupon_id')
    columns_to_normalize.remove('User_id')
    
    columns_to_normalize.remove('ps_discount_mean_every_merchant')
    columns_to_normalize.remove('ps_discount_min_every_merchant')
    columns_to_normalize.remove('ps_discount_max_every_merchant')
    
    columns_to_normalize.remove('ps_distance_mean_every_merchant')
    columns_to_normalize.remove('ps_distance_min_every_merchant')
    columns_to_normalize.remove('ps_distance_max_every_merchant')
    
    return merchant_feature,coupon_feature,user_feature


#获取趋势度量特征
def get_tendency_feature(df):
    #获取用卷消费样本
    data_ps=get_data_ps(df)
    
    #被使用优惠卷的优惠率均值/最小值/最大值/中位数
    rate_mean=data_ps['discount_rate'].mean()
    rate_min=data_ps['discount_rate'].min()
    rate_max=data_ps['discount_rate'].max()
    rate_median=data_ps['discount_rate'].median()
    
    #用户用卷消费距离均值/中位数/最小值/最大值
    distance_mean=float(round(data_ps['Distance'].mean()))
    distance_median=data_ps['Distance'].median()
    distance_min=data_ps['Distance'].min()
    distance_max=data_ps['Distance'].max()
    
    #特征数据字典
    tendency_feature={'rate_mean' : rate_mean,
          'rate_min' : rate_min,
          'rate_max' : rate_max,
          'rate_median' : rate_median,
          'distance_mean' : distance_mean,
          'distance_median' : distance_median,
          'distance_min' : distance_min,
          'distance_max' : distance_max
        }
    
    return tendency_feature


#获取日期特征
def get_days_feature(df):
    days_feature = df.copy()
    
    # 是否为周末
    days_feature['is_weekend'] = days_feature['Date_received'].map(lambda x: 1 if x.weekday() == 5 
                                                                   or x.weekday() == 6 
                                                                   else 0) 
    
    
    return days_feature


#获取线上特征
def get_online_feature(df):
    data = df.copy()
    
    #特征数据集
    merchant_feature=data['Merchant_id'].copy()  #商家特征
    coupon_feature=data['Coupon_id'].copy()      #优惠卷特征
    user_feature=data['User_id'].copy()          #用户特征
    
    
    #用户线上特征
    
    #各用户 线上消费数
    f=data[data['Date'].notnull()]
    f['buy_online_num_every_user']=1
    f=f[['User_id','buy_online_num_every_user']]
    pivot=pd.DataFrame(
        pd.pivot_table(f,index=['User_id'],
                       values=['buy_online_num_every_user'],
                       aggfunc=len)).reset_index()
    user_feature=pd.merge(user_feature, pivot,on='User_id',how='left')
    
    #各用户 线上用卷消费数
    f=get_data_ps(data)
    f['ps_online_num_every_user']=1
    f=f[['User_id','ps_online_num_every_user']]
    pivot=pd.DataFrame(
        pd.pivot_table(f,index=['User_id'],
                       values=['ps_online_num_every_user'],
                       aggfunc=len)).reset_index()
    user_feature=pd.merge(user_feature, pivot,on='User_id',how='left')
    
    
    columns_to_normalize.extend(user_feature.columns.tolist())
    columns_to_normalize.remove('User_id')
    
    
    return user_feature,merchant_feature,coupon_feature


#构造特征数据集
def get_dataset(feature,target,online):
    # 特征工程
    merchant_feature,coupon_feature,user_feature=get_main_feature(feature)   #主要特征
    user_online,merchant_online,coupon_online=get_online_feature(online)     #线上特征
    tendency_feature=get_tendency_feature(feature)                           #趋势度量特征
    days_feature=get_days_feature(target)                                    #日期特征
    
    #获取样本集合
    days_feature=get_data_ss(days_feature)
    
    
    #参数
    user_id_values=days_feature['User_id'].values
    merchant_id_values=days_feature['Merchant_id'].values
    coupon_id_values=days_feature['Coupon_id'].values
    
    #添加主要特征
    
    #商家特征
    f=merchant_feature[merchant_feature['Merchant_id'].isin(merchant_id_values)].copy()
    f.drop_duplicates(keep='first', inplace=True)
    dataset=pd.merge(days_feature,f,on='Merchant_id',how='left')
    
    #优惠卷特征
    f=coupon_feature[coupon_feature['Coupon_id'].isin(coupon_id_values)].copy()
    f.drop_duplicates(keep='first', inplace=True)
    dataset=pd.merge(dataset,f,on='Coupon_id',how='left')
    
    #用户特征
    f=user_feature[user_feature['User_id'].isin(user_id_values)].copy()
    f.drop_duplicates(keep='first', inplace=True)
    dataset=pd.merge(dataset,f,on='User_id',how='left')
   
    #添加线上特征
    
    #用户线上特征
    f=user_online[user_online['User_id'].isin(user_id_values)].copy()
    f.drop_duplicates(keep='first', inplace=True)
    dataset=pd.merge(dataset,f,on='User_id',how='left')
    
    
    #添加趋势度量特征
    for key, val in tendency_feature.items():
        dataset.insert(loc=len(dataset.columns), column=key,value=val)
    
    
    # 删除无用属性
    if 'Date' in dataset.columns.tolist():  
        dataset.drop(['Merchant_id', 'Discount_rate', 'Date'], axis=1, inplace=True)
    else:  
        dataset.drop(['Merchant_id', 'Discount_rate'], axis=1, inplace=True)
    
    
    #用均值填补空值
    for i in dataset.columns.tolist():
        if dataset[i].isnull().any():
            mean=dataset[i].mean()
            dataset[i].fillna(value=mean,inplace=True)
    
    
    # 修正数据类型
    dataset['Coupon_id'] = dataset['Coupon_id'].map(int)    
    dataset['Distance'] = dataset['Distance'].map(int)
    dataset['Date_received']=dataset['Date_received'].apply(lambda x: int(x.strftime("%Y%m%d")))
    
    # 去重
    dataset.drop_duplicates(keep='first', inplace=True)
    dataset.index = range(len(dataset))
    
    return dataset
    

def model_xgb(train, test):
    #训练数据特征
    train_feature=train.drop(['User_id', 'Coupon_id', 'Date_received', 'target'], axis=1)
    #训练数据目标
    train_target=train['target']
    #测试数据特征
    test_feature=test.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1)
    
    
    #数据标准化(z分数规范化)
    col=list(set(columns_to_normalize)) #列表去重
    train_feature[col] = train_feature[col].apply(zscore)
    test_feature[col] = test_feature[col].apply(zscore)
    
    
    # xgboost模型初始化设置
    dtrain=xgb.DMatrix(train_feature,train_target)
    dtest=xgb.DMatrix(test_feature)
    watchlist = [(dtrain,'train')]
    
    # xgb参数
    params = {
      'booster':'gbtree',
      'objective': 'binary:logistic',
      'eval_metric':'auc',
      'gamma':0.1,
      'min_child_weight':1.1,
      'max_depth':5,
      'lambda':5,
      'subsample':0.75,
      'colsample_bytree':0.7,
      'colsample_bylevel':0.7,
      'eta': 0.01,
      'tree_method':'exact',
      'seed':0,
      'nthread':12
            }
    
    # 训练
    model = xgb.train(params, dtrain, num_boost_round=100, evals=watchlist)
    
    # 预测
    ypred=model.predict(dtest)
    
    # 结果处理
    ypred = pd.DataFrame(ypred, columns=['prediction'])
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], ypred], axis=1)
    
    # 特征权重
    feature_weight = pd.DataFrame(columns=['feature_name', 'weight'])
    feature_weight['feature_name'] = model.get_score().keys()
    feature_weight['weight'] = model.get_score().values()
    feature_weight.sort_values(['weight'], ascending=False, inplace=True)
    # 返回
    return result, feature_weight


if __name__ == '__main__':
    #读取文件
    offline=pd.read_csv(offline_path)
    online=pd.read_csv(online_path)
    test=pd.read_csv(test_path)
    
    # 预处理
    offline = preprocess(offline)
    online=preprocess(online)
    test = preprocess(test)
    
    # 添加目标值
    offline = add_target(offline)
    
    
    #划分数据集
    train_feature,train_target,verify_feature,verify_target,test_feature=divide_dataset(offline)
    train_online,verify_online,test_online=divide_dataset(online)                          
    test_target=test.copy()                
    
    
    #构造训练特征数据集
    train_set=get_dataset(train_feature,train_target,train_online)
    #构造验证特征数据集
    verify_set=get_dataset(verify_feature,verify_target,verify_online)
    #构造测试特征数据集
    test_set=get_dataset(test_feature,test_target,test_online)
    

    #训练
    big_train = pd.concat([train_set, verify_set], axis=0)
    result, feature_weight = model_xgb(big_train, test_set)
    
    #输出权重
    print(feature_weight)
    
    result.to_csv("D:\\Zdata\\result.csv", index=False, header=None)
    
   
    
