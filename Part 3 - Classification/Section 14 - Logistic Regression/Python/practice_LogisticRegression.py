#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 18:00:49 2020

@author: hemalatha

Logistic Regression
"""

#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import data
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
std_sc = StandardScaler()
x_train_sc = std_sc.fit_transform(x_train)
x_test_sc =std_sc.fit_transform(x_test)

#Training model
from sklearn.linear_model import LogisticRegression
log_reg_obj = LogisticRegression()
log_reg_obj.fit(x_train_sc, y_train)

# Test model
print(log_reg_obj.predict(std_sc.transform([[30,87000]])))

# Predict test data output
predict_op =  log_reg_obj.predict(x_test_sc)
pred_vs_test = np.concatenate((predict_op.reshape(len(predict_op),1), y_test.reshape(len(y_test),1)),1)
print(pred_vs_test)

# Confusion matrix for classification accuracy
from sklearn import metrics
classifier_accuracy_score = metrics.accuracy_score(y_test, predict_op)
print(classifier_accuracy_score)
classifier_confusion_matrix = metrics.confusion_matrix(y_test, predict_op)
print(classifier_confusion_matrix)
