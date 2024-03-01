# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 09:32:25 2024

@author: 86158
"""

#用于分析online数据的代码，根据任务一相应代码修改而来
import pandas as pd
import matplotlib.pyplot as plt

online_path="data\\ccf_online_stage1_train.csv"

data=pd.read_csv(online_path)

#数据观察与绘图
def data_plot(df):
    ds=df.copy()
    
    
    #用户领卷次数饼状图
    
    #按User_id分组统计领卷次数
    user_coupon=ds.groupby(by="User_id")["Coupon_id"].count()
    #按领卷次数分组统计用户数
    user_coupon=user_coupon.groupby(user_coupon).count()
    #处理数据
    coupon1=user_coupon.values
    coupon1_indexs=["once","twice","Three or more times"]
    coupon1_values=[coupon1[0],coupon1[1],sum(coupon1[2:])]
    user_coupon=pd.Series(data=coupon1_values,index=coupon1_indexs,name="user_number")
    #作饼状图
    user_coupon.plot(kind="pie",autopct="%.2f")    
    
    
    #商户发卷能力饼状图
    
    #按Merchant_id分组统计发卷次数
    merchant_coupon=ds.groupby(by="Merchant_id")["Coupon_id"].count()
    #处理数据
    merchant_coupon=merchant_coupon.sort_values()
    coupon2=merchant_coupon.values
    coupon2_indexs=["Other merchants","Top 20 merchants"]
    coupon2_values=[sum(coupon2[0:len(coupon2)-20]),sum(coupon2[len(coupon2)-20:])]
    #作饼状图
    plt.figure()
    plt.pie(coupon2_values,labels=coupon2_indexs,autopct="%.2f")
    plt.title("coupon_number")
    plt.show()
    
#测试数据分析
def test_count(df):
    test=df.copy()
    
    sample_size=test.shape[0]                             #样本数
    user_size=test["User_id"].value_counts().size         #用户数
    merchant_size=test["Merchant_id"].value_counts().size #商户数
    
    coupon_size=test["Coupon_id"].count()                 #优惠券发放数
    coupon_type_size=test["Coupon_id"].value_counts().size#优惠券种类数

    
    received_size=test["Date_received"].count()           #优惠券领取数
    date_received_max=test["Date_received"].max()         #领取优惠卷最晚日期
    date_received_min=test["Date_received"].min()         #领取优惠卷最早日期
    
    
    print("样本数: ",sample_size,"\n")
    print("用户数: ",user_size,"\n")
    print("商户数: ",merchant_size,"\n")
    
    print("优惠券发放数: ",coupon_size)
    print("优惠券种类数: ",coupon_type_size,"\n")
    
    
    print("优惠券领取数: ",received_size)
    print("领取优惠卷最晚日期: ",date_received_max)
    print("领取优惠卷最早日期: ",date_received_min)
    
data_plot(data)
test_count(data)
    
    
    
    
