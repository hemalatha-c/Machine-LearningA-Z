#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:55:01 2020

@author: hemalatha

Kernel PCA
"""
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, 2:-1].values
y = dataset.iloc[:, -1].values

#split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y)

#feature scaling
from sklearn.preprocessing import StandardScaler
std_sc = StandardScaler()
x_train_sc = std_sc.fit_transform(x_train)
x_test_sc = std_sc.transform(x_test)

# Kernel PCA
from sklearn.decomposition import KernelPCA
kernel_pca = KernelPCA(n_components=2, kernel='rbf')
x_train_sc = kernel_pca.fit_transform(x_train_sc)
x_test_sc = kernel_pca.transform(x_test_sc)

#build log classifier
from sklearn.linear_model import LogisticRegression
classifier_logreg = LogisticRegression(random_state=0)
classifier_logreg.fit(x_train_sc, y_train)
y_pred = classifier_logreg.predict(x_test_sc)

# Conf matrix, accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score
conf_matrix = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test, y_pred)