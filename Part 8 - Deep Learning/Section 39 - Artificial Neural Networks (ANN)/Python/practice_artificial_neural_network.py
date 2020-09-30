#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 16:28:49 2020

@author: hemalatha

ANN - Churn modelling - churn is people leaving the bank/company
"""

#import libraries
import numpy as np
import pandas as pd
import tensorflow as tf

#import dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:,-1].values
##################################  DATA PrePROCESSING ##################################
#encoding categorical data
# label encoding gender col
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
x[:,2] = label_encoder.fit_transform(x[:,2])

#one hot encoding country
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

#split data into train test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Feature scaling : mandate for ANN
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
x_train_sc = std_scaler.fit_transform(x_train)
x_test_sc = std_scaler.fit_transform(x_test)

################################## BUILDING ANN ##################################
#Inititalize ANN as sequence of layers
ann = tf.keras.models.Sequential()

#add input layer and first hidden layer
#NOTE: Default input neurons are taken by num of features in dataset
ann.add(tf.keras.layers.Dense(units = 6, activation='relu'))

# add second hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation='relu'))

#add output layer
ann.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))
# sigmoid used for binary classification; 
# softmax used for multi classification
################################## TRAIN ANN ##################################
# compile ann
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics=['accuracy'])

#train ann
ann.fit(x_train_sc, y_train, batch_size=32, epochs=100)

#predict on single customer
#ann.predict(std_scaler.transform([[1,0,0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
# less chance of customer leaving bank

#predict x_test
y_pred = ann.predict(x_test_sc)
y_pred = (y_pred >  0.5)

print(np.concatenate((np.reshape(y_test, (len(y_test),1)), np.reshape(y_pred, (len(y_pred),1))),1))

#confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
print("conf_matrix :\n", confusion_matrix(y_test, y_pred))
print("Accuracy:\t", accuracy_score(y_test,y_pred))