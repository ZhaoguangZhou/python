# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 13:26:18 2024

@author: 86158
"""


import pandas as pd
import numpy as np

off_train_path=""
off_train=pd.read_csv(off_train_path)

user_feature=off_train["User_id"]



#用户领卷数
ds=off_train[off_train["Coupon_id"]!=np.nan]
uf1=ds.groupby(by="User_id").count()
uf1.name="user_received_num"
user_feature=user_feature.merge(user_feature,uf1,on="User_id",how="left")


#用户领卷并消费数
ds=off_train[(off_train["Coupon_id"]!=np.nan)&(off_train["Date"]!=np.nan)]
uf2=ds.groupby(by="User_id").count()
uf2.name="user_received_buy_num"
user_feature=user_feature.merge(user_feature,uf2,on="User_id",how="left")


#用户领卷未消费数
ds=off_train[(off_train["Coupon_id"]!=np.nan)&(off_train["Date"]==np.nan)]
uf3=ds.groupby(by="User_id").count()
uf3.name="user_received_nobuy_num"
user_feature=user_feature.merge(user_feature,uf3,on="User_id",how="left")


#用户领卷并消费数/消费数
user_feature["user_received_buy_rate"]=user_feature["user_received_buy_num"]/user_feature["user_received_num"]


#用户领取并消费优惠卷的平均折扣率
ds=off_train[(off_train["Date_received"]!=np.nan)&(off_train["Discount_rate"]!=np.nan)]
uf5=ds.groupby(by="User_id")["Discount_rate"].mean()
uf5.name="discount_rate_mean"
user_feature=user_feature.merge(user_feature,uf5,on="User_id",how="left")


#用户领取并消费优惠卷的平均距离
ds=off_train[(off_train["Date_received"]!=np.nan)&(off_train["Distance"]!=np.nan)]
uf6=ds.groupby(by="User_id")["Distance"].mean()
uf6.name="distance_mean"
user_feature=user_feature.merge(user_feature,uf6,on="User_id",how="left")


#用户在多少不同商家领取并消费优惠卷
ds=off_train[(off_train["Date_received"]!=np.nan)&(off_train["Date"]!=np.nan)]
uf7=ds.groupby(by=["User_id","Merchant_id"]).count()
uf7.name="user_received_buy_merchant_num"
user_feature=user_feature.merge(user_feature,uf7,on=["User_id","Merchant_id"],how="left")


#用户在多少不同商家领取优惠卷
ds=off_train[off_train["Date_received"]!=np.nan]
uf8=ds.groupby(by=["User_id","Merchant_id"]).count()
uf8.name="user_received_merchant_num"
user_feature=user_feature.merge(user_feature,uf8,on=["User_id","Merchant_id"],how="left")


#用户在多少不同商家领取并消费优惠卷/用户在多少不同商家领取优惠卷
user_feature["user_received_buy_merchant_rate"]=user_feature["user_received_buy_merchant_num"]/user_feature["user_received_merchant_num"]



