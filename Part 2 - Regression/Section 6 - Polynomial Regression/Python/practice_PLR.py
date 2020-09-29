#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:58:43 2020

@author: hemalatha
Plynomial Linear Regression
"""

#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import data
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

# Plonomial lin reg = lin reg model+ ploy reg model  
#Train liner reg model
from sklearn.linear_model import LinearRegression
lin_reg_obj = LinearRegression()
lin_model = lin_reg_obj.fit(X, Y)

# Build polynomial reg model i.e., for b2 co-eff
from sklearn.preprocessing import PolynomialFeatures
poly_feature = PolynomialFeatures(degree= 4)
x_poly = poly_feature.fit_transform(X)
# Build lin reg model with degree 2 feature
lin_reg_obj_2 = LinearRegression()
lin_reg_model_2 = lin_reg_obj_2.fit(x_poly, Y)


# Visualization of Linear reg
plt.scatter(X, Y, color="red")
plt.plot(X,lin_model.predict(X),color="blue")
plt.title("Salary Position authentication (Linear Reg.)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

#Visualization of polynomial reg
plt.scatter(X, Y, color="red")
plt.plot(X,lin_reg_model_2.predict(x_poly),color="blue")
plt.title("Salary Position authentication (Poly Linear Reg.)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# Predict salary at position levl 6.5??
salary_6_5 = lin_reg_model_2.predict(poly_feature.fit_transform([[6.5]]))
print(salary_6_5)