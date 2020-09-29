#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 16:51:52 2020

@author: hemalatha

random forest for classification
"""
# import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state =0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)

# train random forest
from sklearn.ensemble import RandomForestClassifier
rand_forest_classifier = RandomForestClassifier(random_state=0)
rand_forest_classifier.fit(x_train_sc, y_train)

# predict
y_pred = rand_forest_classifier.predict(x_test_sc)

# confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:\t", accuracy_score(y_test, y_pred))