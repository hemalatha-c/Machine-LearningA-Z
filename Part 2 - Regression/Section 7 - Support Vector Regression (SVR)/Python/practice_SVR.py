#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 16:14:32 2020

@author: hemalatha

Support Vector Regression

Scenario: To predict the salary of a person being interviewd for a new
position of level 6+, to check if he's truthful or not.

"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import data
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

# Reshape Y to have same shape as that of X, to do feature scaling
Y = np.reshape(Y, (len(Y),1))

# Feature scaling is required, as the feature values gets negligible with that of salary values
from sklearn.preprocessing import StandardScaler

#Use diff scalers for diff feature(i.e. levels, salary), as they're in diff range
std_sc_x = StandardScaler()
std_sc_y = StandardScaler()
# scaling features
x = std_sc_x.fit_transform(X)
y = std_sc_y.fit_transform(Y)

#build SVR model
from sklearn.svm import SVR
svr_obj = SVR(kernel='rbf')
svr_model = svr_obj.fit(x, y )

# predict salary for level 6.5?
# since the SVR is trained with scaled values, prediction to be made on transformed input values only
pred_inv_salary = svr_model.predict(std_sc_x.transform([[6.5]]))
# Since above salary is in transformed way, actual salary value is obtained by inverse tranforming
pred_original_salary = std_sc_y.inverse_transform(pred_inv_salary)

# visualization
plt.scatter(X, Y, color="red")
plt.plot(X, std_sc_y.inverse_transform(svr_model.predict(x)), color="blue")
plt.title("Position Salaries with SVR")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

#high resolution smoother curve
x_grid = np.arange(min(X), max(X), 0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(X, Y, color="red")
plt.plot(x_grid, std_sc_y.inverse_transform(svr_model.predict(std_sc_x.fit_transform(x_grid))), color="blue")
plt.title("High resolution Position Salaries with SVR")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

