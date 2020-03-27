#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 19:11:49 2020

@author: sougata
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV    


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
'''
y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

svc=SVC(kernel='rbf',gamma=0.1,C=1e3)
svc.fit(X,y)


model = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=1)
model.fit(X,y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
print(model.score(X,y))
print(svc.score(X,y))

'''
# save PassengerId for final submission
passengerId = test.PassengerId

# merge train and test
titanic = train.append(test, ignore_index=True)
# create indexes to separate data later on
train_index = len(train)
test_index = len(titanic) - len(test)

titanic.Age=titanic.Age.fillna(np.mean(titanic.Age))
titanic.Cabin=titanic.Cabin.fillna("U")
titanic.Fare=titanic.Fare.fillna(np.median(titanic.Fare))
titanic.Embarked=titanic.Embarked.fillna("S")
titanic.fillna(-999999)
#titanic.drop(['Fare'],axis=1)

titanic['FamilySize']=titanic.Parch+titanic.SibSp+1

titanic.Sex=titanic.Sex.map({'male':0,'female':1})
# create dummy variables for categorical features
pclass_dummies = pd.get_dummies(titanic.Pclass, prefix="Pclass")
cabin_dummies = pd.get_dummies(titanic.Cabin, prefix="Cabin")
embarked_dummies = pd.get_dummies(titanic.Embarked, prefix="Embarked")

titanic_dummies = pd.concat([titanic, pclass_dummies,cabin_dummies, embarked_dummies], axis=1)
titanic_dummies.drop(['Pclass','Cabin','Name','Embarked','Ticket'],axis=1,inplace=True)


train=titanic_dummies[:train_index]
test=titanic_dummies[test_index:]

train.Survived=train.Survived.astype('int')


x=np.array(train.drop(['Survived'],axis=1))
y=np.array(train.iloc[:,1])

x_test=(test.drop(['Survived'],axis=1))
#x_test = x_test.dropna()

#x_test = x_test.reset_index()
x_test.fillna(999, inplace=True)

# create param grid object 
forrest_params = dict(     
    max_depth = [n for n in range(9, 14)],     
    min_samples_split = [n for n in range(4, 11)], 
    min_samples_leaf = [n for n in range(2, 5)],     
    n_estimators = [n for n in range(10, 60, 10)],
)

forest=RandomForestClassifier(n_estimators=100)
# build and fit model 
forest_cv = GridSearchCV(estimator=forest,     param_grid=forrest_params, cv=5) 
forest_cv.fit(x, y)
#print(x_test.isnull())
# random forrest prediction on test set
forest_pred = forest_cv.predict(x_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': forest_pred})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
#print(model.score(X,y))
#print(forest_cv.score(x,y))
