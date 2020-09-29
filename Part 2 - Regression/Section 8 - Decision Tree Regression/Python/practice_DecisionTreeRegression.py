#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 17:50:18 2020

@author: hemalatha

Decision Tree Regression
"""

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import data
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:,-1].values

# Training Decision tree regression
from sklearn.tree import DecisionTreeRegressor
dtree_obj = DecisionTreeRegressor(random_state=0)
dtree_obj.fit(X, Y)

# Predict salary at level 6.5?
predicted_salary = dtree_obj.predict([[6.5]])

#Visualization in high resolution
x_grid = np.arange(min(X), max(X), 0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(X, Y, color="red")
plt.plot(x_grid, dtree_obj.predict(x_grid), color="blue")
plt.title("Position Salaries with Decision Tree")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()