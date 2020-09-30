#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:56:23 2020

@author: hemalatha

XGBoost
"""
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2, random_state = 0)

#build xgboost classifier
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(x_train, y_train)

#logreg
from sklearn.linear_model import LogisticRegression
classifier_logreg = LogisticRegression(random_state=0)
classifier_logreg.fit(x_train, y_train)

#CatBoost
# Training CatBoost on the Training set
from catboost import CatBoostClassifier
classifier_cat = CatBoostClassifier()
classifier_cat.fit(x_train, y_train)

#predict
y_pred = xgb.predict(x_test)
y_pred_logreg = classifier_logreg.predict(x_test)

#conf matrix
from sklearn.metrics import accuracy_score, confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

conf_matrix_logreg = confusion_matrix(y_test, y_pred_logreg)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)

#Model selection with K fold cross validation
from sklearn.model_selection import cross_val_score
cross_val_score_xgb = cross_val_score(estimator = xgb, X=x_train, y=y_train,cv=10)
cross_val_score_logreg = cross_val_score(estimator = classifier_logreg, X=x_train, y=y_train,cv=10)

print("Accuracy with XGB: {:.2f}%".format(cross_val_score_xgb.mean()*100))
print("STD with XGB: {:.2f}%".format(cross_val_score_xgb.std()*100))
print("\n\nAccuracy with LogReg: {:.2f}%".format(cross_val_score_logreg.mean()*100))
print("STD with LogReg: {:.2f}%".format(cross_val_score_logreg.std()*100))
