#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 11:25:50 2020

@author: hemalatha

Breast Cancer Detection with Logistic Regression

"""

# Import library
import pandas as pd

# Import data from UCI repository
dataset = pd.read_csv('breast_cancer.csv')
X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,-1].values

# Split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Train Logistic Regression
from sklearn.linear_model import LogisticRegression
log_classifier = LogisticRegression()
log_classifier.fit(x_train, y_train)

# predict
y_pred = log_classifier.predict(x_test)

# Confusion matrix and accuracy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

conf_matrix = confusion_matrix(y_test, y_pred)
print("CONFUSION matrix\n",conf_matrix)
print("Accuracy\n", accuracy_score(y_test, y_pred))

# k-fold cross validation matrix
from sklearn.model_selection import cross_val_score
cross_val_avg = cross_val_score(estimator = log_classifier, X=x_train, y=y_train, cv = 10)
print("Accuracy: {:.2f}%".format(cross_val_avg.mean()*100))
print("Std Deviation: {:.2f}%".format(cross_val_avg.std()*100))