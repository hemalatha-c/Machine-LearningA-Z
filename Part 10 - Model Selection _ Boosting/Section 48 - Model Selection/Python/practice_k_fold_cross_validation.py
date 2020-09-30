#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 14:23:26 2020

@author: hemalatha
k-fold cross validation for model selection 

"""

#import libraries
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt

#import data
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, 2:-1].values
y = dataset.iloc[:, -1].values

#split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
                                                    random_state =0)

#feature scaling
from sklearn.preprocessing import StandardScaler
std_sc = StandardScaler()
x_train_sc = std_sc.fit_transform(x_train)
x_test_sc = std_sc.transform(x_test)

#build a classifier
from sklearn.linear_model import LogisticRegression
classifier_logreg = LogisticRegression(random_state=0)
classifier_logreg.fit(x_train_sc, y_train)
y_pred = classifier_logreg.predict(x_test_sc)

#svm
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train_sc, y_train)
y_pred_2 = svc.predict(x_test_sc)

#conf matrix and accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score
conf_matrix = confusion_matrix(y_test, y_pred_2)
accuracy = accuracy_score(y_test, y_pred_2)

#k-fold validation
from sklearn.model_selection import cross_val_score
accuracies_svc = cross_val_score(estimator= svc, X= x_train_sc, y = y_train, cv=10)
accuracies_logreg = cross_val_score(estimator= classifier_logreg, X= x_train_sc, y = y_train, cv=10)
print("Accuracy for SVC: {:.2f}%".format(accuracies_svc.mean()*100))
print("Accuracy for LogReg: {:.2f}%".format(accuracies_logreg.mean()*100))