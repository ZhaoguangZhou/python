# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 09:01:50 2024

@author: 86158
"""

import pandas as pd

def preprocessing(df):
    ds=df.copy()
    
    print("数据缺失率： ",ds.isnull().sum()/ds.shape[0])
    #处理User_id、Merchant_id和Coupon_id,转换为int类型
    ds["User_id"]=ds["User_id"].apply(int)
    ds["Merchant_id"]=ds["Merchant_id"].apply(int)
    ds["Coupon_id"]=ds["Coupon_id"].apply(int)
    #处理Discount_rate,消费卷类型判断(off满减)
    ds["is_off"]=ds["Discount_rate"].apply(
        lambda x:1 if ':' in str(x) else 0
        )
    #增加min_spend列,对满减类型卷给出消费最低价格，折扣卷赋值-1
    ds["min_spend"]=ds["Discount_rate"].apply(
        lambda x: -1 if ':' not in str(x) else str(x).split(':')[0]
        )
    #增加cut_money列,满减型消费卷，给出优惠额度
    ds["cut_money"]=ds["Discount_rate"].apply(
        lambda x: 0 if ':' not in str(x) else str(x).split(':')[1]
        )
    #处理Discount_rate,将消费卷优惠率统一转换为折扣率
    ds["Discount_rate"]=ds["Discount_rate"].apply(
        lambda r:float(r) if ':' not in r else 
        (float(str(r).split(':')[0])-float(str(r).split(':')[1]))/float(str(r).split(':')[0])
        )
    #处理Distance,将空值替换为-1
    ds["Distance"].fillna(value=-1,inplace=True)
    #处理Date_received,转换为标准时间戳
    ds["Date_received"]=pd.to_datetime(ds["Date_received"],format='%Y%m%d')
    
    return ds;





if __name__ == '__main__':
    off_test_path="data\\ccf_offline_stage1_test_revised.csv"
    off_test=pd.read_csv(off_test_path)
    print("处理之后的数据:\n",preprocessing(off_test));
    