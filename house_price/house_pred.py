#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 21:34:45 2020

@author: sougata
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as  sns

train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')

data=train_data.append(test_data,ignore_index=True)

predict_id=test_data.Id
train_index=len(train_data)
test_index=len(data)-len(test_data)

del data['Alley']
del data['FireplaceQu']
del data['PoolQC']
del data['Fence']
del data['MiscFeature']

data['LotFrontage']=data['LotFrontage'].fillna(data['LotFrontage'].mean())
data['MasVnrArea']=data['MasVnrArea'].fillna(data['MasVnrArea'].mean())
data['BsmtFinSF1']=data['BsmtFinSF1'].fillna(data['BsmtFinSF1'].mean())
data['BsmtFinSF2']=data['BsmtFinSF2'].fillna(data['BsmtFinSF2'].mean())
data['BsmtUnfSF']=data['BsmtUnfSF'].fillna(data['BsmtUnfSF'].mean())
data['TotalBsmtSF']=data['TotalBsmtSF'].fillna(data['TotalBsmtSF'].mean())
data['BsmtFullBath']=data['BsmtFullBath'].fillna(data['BsmtFullBath'].mean())
data['BsmtHalfBath']=data['BsmtHalfBath'].fillna(data['BsmtHalfBath'].mean())
data['GarageYrBlt']=data['GarageYrBlt'].fillna(data['GarageYrBlt'].mean())
data['GarageCars']=data['GarageCars'].fillna(data['GarageCars'].mean())
data['GarageArea']=data['GarageArea'].fillna(data['GarageArea'].mean())


null_catagorical=['MSZoning','Utilities','Exterior1st','Exterior2nd','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','SaleType']
for catagorical in null_catagorical:
    data[catagorical]=data[catagorical].fillna(data[catagorical].mode()[0])

    
catagorical_feature=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','PavedDrive','GarageCond','SaleType','SaleCondition']
#for catagorical in catagorical_feature:
#    catagorical= pd.get_dummies(data[catagorical], prefix=catagorical)
#    data_dummy=pd.concat([catagorical],axis=1)

def category_onhot_multcols(multcol):
    data_final=data
    i=0
    for field in multcol:
        df1=pd.get_dummies(data[field],drop_first=True)
        data.drop([field],axis=1,inplace=True)
        if i==0:
            data_final=df1.copy()
        else:
            data_final=pd.concat([data_final,df1],axis=1)
        i+=1
    data_final=pd.concat([data,data_final],axis=1)
    return data_final

main_data=data.copy
final_data=category_onhot_multcols(catagorical_feature)


final_data=final_data.loc[:,~final_data.columns.duplicated()]

train=final_data[:train_index]
train.dropna(inplace=True)
test=final_data[test_index:]
del test['SalePrice']
#test.dropna(inplace=True)

x_train=train.drop(['SalePrice'],axis=1)
y_train=train['SalePrice']

from sklearn.ensemble import RandomForestRegressor 
classifier=RandomForestRegressor()
classifier.fit(x_train,y_train)

prediction=classifier.predict(test)

#print(classifier.score(x_train,y_train))










