#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 18:05:50 2020

@author: hemalatha

PCA - Principal Component Analysis
"""

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import data
dataset = pd.read_csv('Wine.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2,
                                                    random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
std_sc = StandardScaler()
x_train_sc = std_sc.fit_transform(x_train)
x_test_sc = std_sc.transform(x_test)


#apply PCA
from sklearn.decomposition.pca import PCA
#no of components/features to retain at end
pca_obj = PCA(n_components=2)
x_train_sc = pca_obj.fit_transform(x_train_sc)
x_test_sc = pca_obj.transform(x_test_sc)


#build a classifier
from sklearn.linear_model.logistic import LogisticRegression
log_reg_classifier = LogisticRegression(random_state=0)
log_reg_classifier.fit(x_train_sc, y_train)
y_pred= log_reg_classifier.predict(x_test_sc)

#performance metrics
from sklearn.metrics import confusion_matrix, accuracy_score
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)