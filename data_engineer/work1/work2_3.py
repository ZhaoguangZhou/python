# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 18:55:35 2024

@author: 86158
"""

import pandas as pd
import numpy as np

#随机生成十行五列数据
arr=np.random.rand(10,5)   

#调整数据类型和范围，便于操作
arr=arr*10
arr=arr.astype(int)        

#设置行列标签
row_tag=["day{}".format(i) for i in range(1,11)]                 
col_tag=["00:00","07:00","12:00","14:00","20:00"]

#生成Dataframe
df=pd.DataFrame(data=arr,index=row_tag,columns=col_tag)

#输出df数据
print(df)

#绘制柱状图
ax=df.plot(kind="bar")
ax.set_xlabel("date")
ax.set_ylabel("temperature")

#绘制散点图
ax=df.plot(x="00:00",y="12:00",kind="scatter")
ax.set_xlabel("00:00")
ax.set_ylabel("12:00")

#绘制折线图
ax=df.plot(title="10-day temperature change chart")
ax.set_xlabel("date")
ax.set_ylabel("temperature")

#绘制箱线图
ax=df.plot.box(title="Temperature box")
ax.set_xlabel("time")
ax.set_ylabel("temperature")

#绘制区域图（面积图）
ax=df.plot(kind="area")
ax.set_xlabel("date")
ax.set_ylabel("temperature area")
