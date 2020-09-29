#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 23:31:42 2020

@author: hemalatha

SVM
"""
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import data
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, : -1].values
y = dataset.iloc[:, -1].values

# Split data into train-test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x ,y, test_size = 0.25, random_state =0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)

# Train SVM
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(x_train_sc, y_train)
y_pred = svc.predict(x_test_sc)

y_test_pred = np.concatenate((np.reshape(y_test,(len(y_test),1)), np.reshape(y_pred,(len(y_pred),1))),1)

# Confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
svm_accuracy_score = accuracy_score(y_test, y_pred)
print(svm_accuracy_score)