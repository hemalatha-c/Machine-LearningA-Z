#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:17:44 2020

@author: hemalatha

Random Forest Regressor
"""

#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import data
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

#Train random forest
from sklearn.ensemble import RandomForestRegressor
random_forest_obj = RandomForestRegressor(n_estimators=10, random_state=0)
random_forest_obj.fit(X, Y)

#Predict salary for level 6.5?
pred_sal = random_forest_obj.predict([[6.5]])

#Visualization with high resolution for smoother curve
x_grid = np.arange(min(X), max(X), 0.01)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(X, Y, color="red")
plt.plot(X,random_forest_obj.predict(X), color="blue")
plt.title("Position Salaries with RandomForest")
plt.xlabel("Position level")
plt.ylabel("Salaries")
plt.show()