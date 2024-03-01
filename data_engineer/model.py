# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 08:26:49 2024

@author: 86158
"""



import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
from sklearn.model_selection import train_test_split
dataset1 = pd.read_csv('D:\Zdata\ProcessDataSet1.csv')
dataset1.label.replace(-1,0,inplace=True) 
dataset2 = pd.read_csv('D:\Zdata\ProcessDataSet2.csv')
dataset2.label.replace(-1,0,inplace=True)
dataset3 = pd.read_csv('D:\Zdata\ProcessDataSet3.csv')

dataset1.drop_duplicates(inplace=True)
dataset2.drop_duplicates(inplace=True)
dataset12 = pd.concat([dataset1,dataset2],axis=0)
dataset12_y = dataset12.label
dataset12_x = dataset12.drop(['user_id','label','day_gap_before','coupon_id','day_gap_after'],axis=1)      
                                         
dataset3.drop_duplicates(inplace=True)                       
dataset3_preds = dataset3[['user_id','coupon_id','date_received']]
dataset3_x = dataset3.drop(['user_id','coupon_id','date_received','day_gap_before','day_gap_after'],axis=1)

dataTrain = xgb.DMatrix(dataset12_x,label=dataset12_y)
dataTest = xgb.DMatrix(dataset3_x)

#性能评价函数
def myauc(test):
    testgroup = test.groupby(['coupon_id'])
    aucs = []
    for i in testgroup:
        tmpdf = i[1] 
        if len(tmpdf['label'].unique()) != 2:
            continue
        fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred'], pos_label=1)
        aucs.append(auc(fpr,tpr))
    return np.average(aucs)


# In[4]:


params={'booster':'gbtree',
	    'objective': 'rank:pairwise',
	    'eval_metric':'auc',
	    'gamma':0.1,
	    'min_child_weight':1.1,
	    'max_depth':5,
	    'lambda':10,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.01,
	    'tree_method':'exact',
	    'seed':0,
	    'nthread':12
	    }

cvresult = xgb.cv(params, dataTrain, num_boost_round=200, nfold=5, metrics='auc', seed=0, verbose_eval=False, early_stopping_rounds=50)

num_round_best = cvresult.shape[0] - 1
print('Best round num: ', num_round_best)

watchlist = [(dataTrain,'train')]
model1 = xgb.train(params,dataTrain,num_boost_round=num_round_best,evals=watchlist)

model1.save_model('D:/Zdata/xgbmodel1')
print('------------------------train done------------------------------')


# In[5]:


model1=xgb.Booster()

model1.load_model('D:/Zdata/xgbmodel1') 

temp = dataset12[['coupon_id','label']].copy()

temp['pred'] =model1.predict(xgb.DMatrix(dataset12_x))

temp.pred = MinMaxScaler(copy=True,feature_range=(0,1)).fit_transform(temp['pred'].values.reshape(-1,1))

print(myauc(temp))



# In[6]:


dataset3_preds2 = dataset3_preds

dataset3_preds2['label'] = model1.predict(dataTest)

dataset3_preds2['label'] = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(dataset3_preds2['label'].values.reshape(-1, 1))


dataset3_preds2.sort_values(by=['coupon_id','label'],inplace=True)

dataset3_preds2.to_csv("D:/Zdata/xgb_preds2.csv",index=None,header=None)

print(dataset3_preds2.describe())







