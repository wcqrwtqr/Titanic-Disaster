#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 10:40:14 2018

@author: mohammedalbatati
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


# Master Parameters:
n_splits = 5 # Cross Validation Splits
n_iter = 70 # Randomized Search Iterations
scoring = 'accuracy' # Model Selection during Cross-Validation
rstate = 25 # Random State used 
testset_size = 0.30




train_dataset = pd.read_csv('train.csv', index_col = 'PassengerId')
test_dataset = pd.read_csv('test.csv', index_col = 'PassengerId')

# combine the two data sets

survived = train_dataset['Survived'].copy()
train_df = train_dataset.drop('Survived', axis=1).copy()

# save the index of each set
traindex = train_df.index
testdex = test_dataset.index

df = pd.concat([train_df,test_dataset])

# delete the extra data sets 
del train_df , test_dataset, train_dataset

# add extra columns

# New Variables engineering, heavily influenced by:
# Kaggle Source- https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
# How many of family size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
#df = df.drop('Family Size' , axis =1)

#Name length
df['Name_length'] = df['Name'].apply(len) 

# add is alone coloumn with initial zeros
df['IsAlone'] = 0
df.loc[df['FamilySize']==1 , 'IsAlone'] = 1


# Title: (Source)
# Kaggle Source- https://www.kaggle.com/ash316/eda-to-prediction-dietanic

df['Title'] = 0
df['Title'] = df.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations



df['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
  ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr','Mrs'],inplace=True)

# working with age and applying the mean for each group
# locating the empty Age for the Title                       Applying the mean for the Title 
df.loc[(df.Age.isnull())&(df.Title=='Mr'), 'Age'] = df.Age[df.Title == 'Mr'].mean()
df.loc[(df.Age.isnull())&(df.Title=='Mrs'), 'Age'] = df.Age[df.Title == 'Master'].mean()
df.loc[(df.Age.isnull())&(df.Title=='Miss'), 'Age'] = df.Age[df.Title == 'Miss'].mean()
df.loc[(df.Age.isnull())&(df.Title=='Other'), 'Age'] = df.Age[df.Title == 'Other'].mean()
df.loc[(df.Age.isnull())&(df.Title=='Master'), 'Age'] = df.Age[df.Title == 'Master'].mean()
df = df.drop('Name' , axis = 1)

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode().iloc[0])

df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

df['Sex'] = df['Sex'].map({'male':0 , 'female':1})

df['Title'] = df['Title'].map({ 'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master':3, 'Other':4})

df['Embarked'] = df['Embarked'].map({'Q': 0, 'S': 1, 'C': 2}).astype(int)

'''
## helper function to check the Nan values
for i in range(13):
    print(df.iloc[: , i].isnull().value_counts())
''' 

df = df.drop(['Ticket', 'Cabin'] , axis = 1)


pd.concat([df.loc[traindex, :], survived], axis=1).hist()

plt.show()

import seaborn as sns

sns.heatmap(pd.concat([df.loc[traindex, :], survived], axis=1).corr(), annot=True, fmt=".2f")

# scale the columns that needs to be scalled 
from sklearn.preprocessing import StandardScaler
for col in ['Fare','Age','Name_length']:
    transf = df[col].reshape(-1,1)
    scaler = StandardScaler().fit(transf)
    df[col] = scaler.transform(transf)

# split the data again
train_df = df.loc[traindex , :]
train_df['Survived'] = survived 
test_df = df.loc[testdex , :]

del df , col  , survived , transf

# definition of X & y
X = train_df.drop(['Survived'], axis=1)
y = train_df['Survived'].copy()

#spliting 

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import StratifiedShuffleSplit

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                    random_state=rstate,
                                                    stratify = y)
#cv = StratifiedShuffleSplit(n_splits=n_splits ,test_size=0.2, random_state=rstate)

'''
from keras.models import Sequential
from keras.activations import relu , sigmoid 
from keras.layers import Dense , Dropout
from keras.optimizers import Adam, RMSprop , Adagrad
from keras.losses import binary_crossentropy

model = Sequential()
model.add(Dense(100 , activation= relu ,  input_dim = 11))
model.add(Dropout(0.3))
model.add(Dense(200, activation=relu))
model.add(Dropout(0.5))
model.add(Dense(200, activation=relu))
model.add(Dropout(0.5))
model.add(Dense(200, activation=relu))
model.add(Dropout(0.5))
model.add(Dense(200, activation=relu))
model.add(Dropout(0.5))
model.add(Dense(200, activation=relu))
model.add(Dropout(0.3))
model.add(Dense(1, activation=sigmoid))

model.summary()

model.compile(Adagrad(lr=0.01) , loss= 'binary_crossentropy' , metrics=['accuracy'])
history = model.fit(X_train , y_train , batch_size=100, epochs=200, verbose=1, 
                    validation_data=(X_test,y_test))
'''
from sklearn.ensemble import VotingClassifier , RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score

rand_forest_model = RandomForestClassifier(n_estimators=100)
naive_model = GaussianNB()
svc_model = SVC(kernel='rbf') # 'linear'

voting_model = VotingClassifier(estimators=[('naive',naive_model), 
                                            ('svc',svc_model),
                                            ('forest',rand_forest_model)]).fit(X_train, y_train)


y_pred = voting_model.predict(X_test)

scoring_acc = voting_model.score(X_test,y_test)
cross = cross_val_score(voting_model,X,y)


#y_pred = model.predict(test_df)
#
#submission = model.predict(test_df)
#df = pd.DataFrame(submission)
#df['PassengerID'] = test_df.index
#df['Survived'] = df.iloc[: , 0].copy()
#df.drop(axis=1 , columns = 0, inplace = True)
#df = df.set_index('PassengerID')
#
#df.loc[(df.Survived > 0.50),'Survived']= 1
#df.loc[(df.Survived < 0.50),'Survived']= 0


#pd.crosstab(test_df.FamilySize, test_df.Pclass)
#
#
#
#X_train.groupby(['Sex', 'Embarked'])['Age'].cumsum()

#import seaborn as sns
#plt.style.use('fivethirtyeight')
#f,ax=plt.subplots(1,3,figsize=(20,8))
#sns.distplot(test_df[test_df['Pclass']==1].Fare,ax=ax[0])
#ax[0].set_title('Fares in Pclass 1')
#sns.distplot(test_df[test_df['Pclass']==2].Fare,ax=ax[1])
#ax[1].set_title('Fares in Pclass 2')
#sns.distplot(test_df[test_df['Pclass']==3].Fare,ax=ax[2])
#ax[2].set_title('Fares in Pclass 3')
#plt.show()

#test_df['Fare_cut'] = pd.qcut(test_df['Fare'],4)
#test_df = test_df.drop(['Fare-cut'],axis=1)
#
#test_df.Fare_cut.value_counts()



















