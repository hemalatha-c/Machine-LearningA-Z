#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 14:57:26 2020

@author: hemalatha

ANN for Regression using tf-2.2

"""

###################################### DATA PREPROCESSING ################
#Import libraries
import numpy as np
import pandas as pd
import tensorflow as tf

#import dataset
dataset = pd.read_excel('Folds5x2_pp.xlsx')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

###################################### BUILD ANN ################
#initialize ANN to be sequential layers
ann = tf.keras.models.Sequential()

#add input and hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation='relu'))

#add second hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation='relu'))

#add output layer, no actiavtion function for regression,
# sigmoid for classification when 2 classes
ann.add(tf.keras.layers.Dense(units = 1))

###################################### TRAIN ANN ################
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')
ann.fit(x_train, y_train, batch_size=32, epochs=100)

###################################### PREDICT   ################
y_pred = ann.predict(x_test)
y_pred_result = np.concatenate((np.reshape(y_test, (len(y_test),1)), np.reshape(y_pred, (len(y_pred),1))),1)