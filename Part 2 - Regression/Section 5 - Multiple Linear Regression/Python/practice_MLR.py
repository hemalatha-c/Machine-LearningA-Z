#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:43:31 2020

@author: hemalatha

Multiple Linear Regression : Multiple independent vars with one dependent var
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import data
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Encode categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
col_transformer_obj = ColumnTransformer(transformers=[('encode', OneHotEncoder(), [3])], remainder='passthrough')
X1 = col_transformer_obj.fit_transform(X)

# Split data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X1, Y, test_size = 0.2, random_state = 0)

# Feature scaling isn't required in MLR, as the co-efficients associated with vars compensate the variation in feature values.

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg_model = lin_reg.fit(x_train, y_train)

predict = lin_reg_model.predict(x_test)

# Comparison of predicted vs actual values
np.set_printoptions(precision = 2)
print(np.concatenate((predict.reshape(len(predict),1), y_test.reshape(len(y_test),1)),1))

#Question 1: How do I use my multiple linear regression model to make a single prediction, for example the profit of a startup with R&D Spend = 160000, Administration Spend = 130000, Marketing Spend = 300000 and State = California?
print(lin_reg_model.predict([[1, 0, 0, 160000, 130000, 300000]]))
#Question 2: How do I get the final regression equation y = b0 + b1 x1 + b2 x2 + ... with the final values of the coefficients?
print(lin_reg_model.coef_)
print(lin_reg_model.intercept_)
# Answer: [ 8.66e+01 -8.73e+02  7.86e+02  7.73e-01  3.29e-02  3.66e-02]
# 42467.52924854249
# Equation is, Profit=86.6×Dummy State 1−873×Dummy State 2+786×Dummy State 3−0.773×R&D Spend+0.0329×Administration+0.0366×Marketing Spend+42467.53

# visualization
plt.scatter(X1, Y, color="red")