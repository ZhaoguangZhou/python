# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 20:38:43 2024

@author: 86158
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

#测试数据分析
def test_count(df):
    test=df.copy()
    
    sample_size=test.shape[0]                             #样本数
    user_size=test["User_id"].value_counts().size         #用户数
    merchant_size=test["Merchant_id"].value_counts().size #商户数
    
    coupon_size=test["Coupon_id"].count()                 #优惠券发放数
    coupon_type_size=test["Coupon_id"].value_counts().size#优惠券种类数

    distance_max=test["Distance"].max()                   #用户与商户最远距离
    distance_min=test["Distance"].min()                   #用户与商户最近距离
    
    received_size=test["Date_received"].count()           #优惠券领取数
    date_received_max=test["Date_received"].max()         #领取优惠卷最晚日期
    date_received_min=test["Date_received"].min()         #领取优惠卷最早日期
    
    
    print("样本数: ",sample_size,"\n")
    print("用户数: ",user_size,"\n")
    print("商户数: ",merchant_size,"\n")
    
    print("优惠券发放数: ",coupon_size)
    print("优惠券种类数: ",coupon_type_size,"\n")
    
    print("最远距离: ",distance_max)
    print("最近距离: ",distance_min,"\n")
    
    print("优惠券领取数: ",received_size)
    print("领取优惠卷最晚日期: ",date_received_max)
    print("领取优惠卷最早日期: ",date_received_min)

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
    
    
    #不同折扣消费卷数量直方图

    #将Discount_rate转换为统一的优惠率
    rate_coupon=ds["Discount_rate"].apply(
        lambda r:float(r) if ':' not in r else 
        (float(str(r).split(':')[0])-float(str(r).split(':')[1]))/float(str(r).split(':')[0])
        )
    #作直方图
    plt.figure()
    plt.hist(rate_coupon, bins=10)
    plt.xlabel("discount_rate")
    plt.ylabel("coupon_number")
    plt.show()
    
    
    #满减型与非满减型优惠卷占比饼状图
    
    #判断优惠卷是否为满减类型
    off_or_discount=ds["Discount_rate"].str.contains(':')
    #处理数据
    off_or_discount=off_or_discount.groupby(off_or_discount).count()
    #作饼状图
    plt.figure()
    plt.pie(off_or_discount.values,labels=["discount","off"],autopct="%.2f")
    plt.title("coupon_number")
    plt.show()
    
    
    #用户与商户距离直方图

    distance=ds["Distance"].values
    plt.figure()
    plt.hist(distance, bins=10)
    plt.xlabel("distance")
    plt.ylabel("user_number")
    plt.show()
    
    
    #每日消费卷领取数量折线图

    received_daily=ds["Date_received"]
    received_daily=received_daily.groupby(received_daily).count()
    plt.figure()
    plt.plot(received_daily.values)
    plt.xlabel("date")
    plt.ylabel("received_number")
    plt.show()
    
    
    #工作日与周末领卷均值柱状图

    received_mean=ds["Date_received"]
    #统计每日的领卷数
    received_mean=received_mean.groupby(received_mean).count()
    #数据处理
    date=received_mean.index     #日期列表
    weekday_size=0               #工作日天数
    weekend_size=0               #周末天数
    received_weekday=0           #工作日领卷数
    received_weekend=0           #周末领卷数
    for i in date:
        #日期转换为星期
        week=datetime.strptime(str(i), '%Y%m%d').weekday()
        if 0<=week<=4:
            weekday_size+=1
            received_weekday+=received_mean[i]
        else:
            weekend_size+=1
            received_weekend+=received_mean[i]
    weekday_mean=received_weekday/weekday_size  #工作日领卷均值
    weekend_mean=received_weekend/weekend_size  #周末领卷均值
    #作柱状图
    plt.figure()
    plt.bar(x=[0,1],height=[weekday_mean,weekend_mean],width=0.2,tick_label=["weekday","weekend"])
    plt.show()
    


if __name__ == '__main__':
    off_test_path="data\\ccf_offline_stage1_test_revised.csv"
    off_test=pd.read_csv(off_test_path)
    test_count(off_test)
    data_plot(off_test)



