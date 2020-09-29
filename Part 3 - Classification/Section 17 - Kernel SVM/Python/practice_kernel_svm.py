#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 01:52:51 2020

@author: hemalatha

Kernel SVM
"""
# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# IMport data
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y, test_size= 0.25, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)

# Train Kernel SVM
from sklearn.svm import SVC
svc_classifier = SVC(kernel='rbf', random_state = 0)
svc_classifier.fit(x_train_sc, y_train)

# predict
print(svc_classifier.predict(sc.transform([[30, 87000]])))
y_pred = svc_classifier.predict(x_test_sc)

# Confusion matrix and accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:\n", accuracy_score(y_test, y_pred))