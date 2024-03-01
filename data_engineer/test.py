# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:46:40 2024

@author: 86158
"""

#其他测试代码
import pandas as pd


#添加目标值
def add_target(df):
    data = df.copy()
    
    #领券后15天内消费为1,否则为0
    data['target'] = list(map(lambda x, y: 1 if (x - y).total_seconds() / (60 * 60 * 24) <= 15 else 0, data["Date"],
                             data["Date_received"]))
    
    return data



def data_plot(df):
    ds=df.copy()
    
    #每日用卷消费数条形图
    received_daily=ds
    
    received_daily=received_daily.groupby(ds["Date"]).count()
    
    received_daily["date"]=received_daily.index
    
    received_daily.plot(x="date",y="target",kind="bar",figsize=(300,120),rot=90,fontsize=20)
    
    
    

#文件路径
offline_path="data\\ccf_offline_stage1_train.csv"


#读取文件
offline=pd.read_csv(offline_path,parse_dates=["Date_received","Date"])


data=add_target(offline)

data=data[data['target']==1]

data_plot(data)


def preprocess(df):
    data=df.copy();
    
    #获取正样本(领卷并消费) positive sample
    data_ps=data[(data["Coupon_id"].notnull() & data["Date"].notnull())]

    #获取普通样本(未领卷并消费) ordinary sample
    data_os=data[(data["Coupon_id"].isnull() & data["Date"].notnull())]

    #获取负样本(领卷未消费) negative sample
    data_ns=data[(data["Coupon_id"].notnull() & data["Date"].isnull())]
    
    print("正样本数量： ",data_ps.shape[0])
    print("普通样本数量： ",data_os.shape[0])
    print("负样本数量： ",data_ns.shape[0])
    