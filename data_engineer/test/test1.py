# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 20:05:38 2024

@author: 86158
"""

#每日用卷消费数条形图绘制代码，根据任务一相应代码修改而来
import pandas as pd

#添加目标值
def add_target(df):
    data = df.copy()
    
    #领券后15天内消费为1,否则为0
    data['target'] = list(map(lambda x, y: 1 if (x - y).total_seconds() / (60 * 60 * 24) <= 15 else 0, data["Date"],
                             data["Date_received"]))
    
    return data

    
    
    

#文件路径
offline_path="data\\ccf_offline_stage1_train.csv"


#读取文件
offline=pd.read_csv(offline_path,parse_dates=["Date_received","Date"])


data=add_target(offline)

ds=data[data['target']==1]

data=data.groupby(data["Date_received"]).count()


#每日用卷消费数条形图
received_daily=ds


received_daily=received_daily.groupby(ds["Date"]).count()


received_daily["date"]=received_daily.index

received_daily["target"]=data["target"]/received_daily["target"]

received_daily.plot(x="date",y="target",kind="bar",figsize=(300,120),rot=90,fontsize=20)
