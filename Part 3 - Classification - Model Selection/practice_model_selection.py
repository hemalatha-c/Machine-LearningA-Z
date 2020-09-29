#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:17:10 2020

@author: hemalatha

Model Selection for classification
"""

# import ibraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[: , -1].values

# split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state =0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)

# train Model
## Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier_log_reg = LogisticRegression()
classifier_log_reg.fit(x_train_sc, y_train)

y_pred_log_reg = classifier_log_reg.predict(x_test_sc)

## KNN
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier()
classifier_knn.fit(x_train_sc, y_train)

y_pred_knn = classifier_knn.predict(x_test_sc)

## SVM
from sklearn.svm import SVC
classifier_svc = SVC()
classifier_svc.fit(x_train_sc, y_train)

y_pred_svc = classifier_svc.predict(x_test_sc)


## decision trees
from sklearn.tree import DecisionTreeClassifier
classifier_decision_tree = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
classifier_decision_tree.fit(x_train_sc, y_train)

y_pred_decision_tree = classifier_decision_tree.predict(x_test_sc)

## naive bayes
from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(x_train_sc, y_train)

y_pred_nb = classifier_nb.predict(x_test_sc)

## random forest
from sklearn.ensemble import RandomForestClassifier
classifier_random_forest = RandomForestClassifier()
classifier_random_forest.fit(x_train_sc, y_train)

y_pred_rand_forest = classifier_decision_tree.predict(x_test_sc)

## Accuracy of models
from sklearn.metrics import accuracy_score
print("Accuracy list are as follows:\n1. Logistic: ", accuracy_score(y_test, y_pred_log_reg))
print("2. KNN: ", accuracy_score(y_test, y_pred_knn))
print("3. SVM: ", accuracy_score(y_test, y_pred_svc))
print("4. Decision Tree: ", accuracy_score(y_test, y_pred_decision_tree))
print("5. NB: ", accuracy_score(y_test, y_pred_nb))
print("6. Random Forest: ", accuracy_score(y_test, y_pred_rand_forest))