#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:15:11 2020

@author: hemalatha

Linear Discriminant Analysis -LDA
"""

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import data
dataset = pd.read_csv('Wine.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#split data 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y)

#feature scaling
from sklearn.preprocessing import StandardScaler
std_sc = StandardScaler()
x_train_sc = std_sc.fit_transform(x_train)
x_test_sc = std_sc.transform(x_test)

#Dimensionality reduction- LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit_transform(x_train_sc, y_train)
lda.transform(x_test_sc)


#build classification model
from sklearn.linear_model import LogisticRegression
classifier_logreg = LogisticRegression(random_state=0)
classifier_logreg.fit(x_train_sc, y_train)
y_pred =  classifier_logreg.predict(x_test_sc)

#confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

