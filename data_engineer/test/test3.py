# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:18:55 2024

@author: 86158
"""

#该代码为部分特征提取测试代码
import pandas as pd

#文件路径
offline_path="data\\ccf_offline_stage1_train.csv"
test_path="data\\ccf_offline_stage1_test_revised.csv"


#读取文件，并将时间转换为datetime类型
offline=pd.read_csv(offline_path,parse_dates=["Date_received","Date"])

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

#获取主要特征
def get_test1(df):
    data = df.copy()
    
    # 返回的特征数据集
    main_feature = data.copy()

    # 各商家用户用卷数
    f1=get_data_ps(data)
    f1['data_ps_num_every_merchant']=1
    f1=f1[['Merchant_id','data_ps_num_every_merchant']]
    pivot=pd.DataFrame(
        pd.pivot_table(f1,index=['Merchant_id'],
                       values=['data_ps_num_every_merchant'],
                       aggfunc=len)).reset_index()
    main_feature=pd.merge(main_feature, pivot,on='Merchant_id',how='left')

    

    # 返回
    return main_feature

def get_test2(df):
    data = df.copy()
    
    # 返回的特征数据集
    main_feature = data.copy()

    #各商家用户购物数
    f2=data[data['Date'].notnull()]
    f2['buy_num_every_merchant']=1
    f2=f2[['Merchant_id','buy_num_every_merchant']]
    pivot=pd.DataFrame(
        pd.pivot_table(f2,index=['Merchant_id'],
                       values=['buy_num_every_merchant'],
                       aggfunc=len)).reset_index()
    main_feature=pd.merge(main_feature, pivot,on='Merchant_id',how='left')

    return main_feature

def get_test3(df):
    data = df.copy()
    
    # 返回的特征数据集
    main_feature = data.copy()
    
    # 各商家用户用卷数
    f1=get_data_ps(data)
    f1['ps_num_every_merchant']=1
    f1=f1[['Merchant_id','ps_num_every_merchant']]
    pivot=pd.DataFrame(
        pd.pivot_table(f1,index=['Merchant_id'],
                       values=['ps_num_every_merchant'],
                       aggfunc=len)).reset_index()
    main_feature=pd.merge(main_feature, pivot,on='Merchant_id',how='left')
    
    #各商家 用户用卷数/发卷数
    f3=get_data_ss(data)
    f3['ps_rate_every_merchant']=1
    f3=f3[['Merchant_id','ps_rate_every_merchant']]
    pivot=pd.DataFrame(
        pd.pivot_table(f3,index=['Merchant_id'],
                       values=['ps_rate_every_merchant'],
                       aggfunc=len)).reset_index()
    main_feature=pd.merge(main_feature, pivot,on='Merchant_id',how='left')
    main_feature['ps_rate_every_merchant']=main_feature['ps_num_every_merchant']/main_feature['ps_rate_every_merchant']
    
    return main_feature

def get_test4(df):
    data = df.copy()
    
    # 返回的特征数据集
    main_feature = data.copy()
    
    #各优惠卷使用次数
    f4=get_data_ps(data)
    f4['used_num_every_coupon']=1
    f4=f4[['Coupon_id','used_num_every_coupon']]
    pivot=pd.DataFrame(
        pd.pivot_table(f4,index=['Coupon_id'],
                       values=['used_num_every_coupon'],
                       aggfunc=len)).reset_index()
    main_feature=pd.merge(main_feature, pivot,on='Coupon_id',how='left')
    
    #各优惠卷 使用次数/发卷数
    f5=get_data_ss(data)
    f5['used_rate_every_coupon']=1
    f5=f5[['Coupon_id','used_rate_every_coupon']]
    pivot=pd.DataFrame(
        pd.pivot_table(f5,index=['Coupon_id'],
                       values=['used_rate_every_coupon'],
                       aggfunc=len)).reset_index()
    main_feature=pd.merge(main_feature, pivot,on='Coupon_id',how='left')
    main_feature['used_rate_every_coupon']=main_feature['used_num_every_coupon']/main_feature['used_rate_every_coupon']
    
    return main_feature




