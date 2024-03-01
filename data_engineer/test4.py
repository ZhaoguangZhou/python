# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:19:03 2024

@author: 86158
"""

#该代码为部分特征提取测试代码
import pandas as pd

#文件路径
offline_path="D:\ApplicationFiles\python\Project1\code1\data\ccf_offline_stage1_train.csv"
test_path="data\\ccf_offline_stage1_test_revised.csv"


#读取文件，并将时间转换为datetime类型
offline=pd.read_csv(offline_path,parse_dates=["Date_received","Date"])


#数据预处理
def preprocess(df):
    data = df.copy()
    
    
    # Discount_rate处理
    
    #消费卷类型判断(off满减)
    data["is_off"]=data["Discount_rate"].apply(
        lambda r:1 if ':' in str(r) else 0
        )
    
    #增加min_cost列,对满减类型卷给出消费最低价格，折扣卷赋值-1
    data["min_cost"]=data["Discount_rate"].apply(
        lambda r: -1 if ':' not in str(r) else float(str(r).split(':')[0])
        )
    
    #将Discount_rate转换为折扣率
    data["discount_rate"]=data[data["Discount_rate"].notnull()]["Discount_rate"].apply(
        lambda r:float(r) if ':' not in r else 
        (float(str(r).split(':')[0])-float(str(r).split(':')[1]))/float(str(r).split(':')[0])
        )
    
    # Distance处理
    
    #将空值替换为均值
    if "Distance" in data.columns:
        distance_mean=data["Distance"].sum()/data[data["Distance"].notnull()].shape[0]
        distance_mean=float(round(distance_mean))
        data["Distance"].fillna(value=distance_mean,inplace=True)
    
    
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


#获取趋势度量特征
def get_tendency_feature(df):
    data = df.copy()
    
    # 返回的特征数据集
    tendency_feature = data.copy()
    
    data_ps=get_data_ps(data)
    
    #被使用优惠卷的优惠率均值/最小值/最大值/中位数
    rate_mean=data_ps['discount_rate'].mean()
    rate_min=data_ps['discount_rate'].min()
    rate_max=data_ps['discount_rate'].max()
    rate_median=data_ps['discount_rate'].median()
    
    #用户用卷消费距离均值/中位数/众数
    distance_mean=float(round(data_ps['Distance'].mean()))
    distance_median=data_ps['Distance'].median()
    distance_mode=data_ps['Distance'].mode()
    
    tendency_feature['rate_mean']=rate_mean
    tendency_feature['rate_min']=rate_min
    tendency_feature['rate_max']=rate_max
    tendency_feature['rate_median']=rate_median
    tendency_feature['distance_mean']=distance_mean
    tendency_feature['distance_median']=distance_median
    tendency_feature['distance_mode']=distance_mode
    
    return tendency_feature


offline=preprocess(offline)

print(get_tendency_feature(offline).shape)