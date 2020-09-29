#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:43:45 2020

@author: hemalatha
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# Split data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state= 0)

# Build linear reg model
from sklearn.linear_model import LinearRegression
regressor_obj = LinearRegression()
regressor_obj.fit(x_train, y_train)

# Predict
y_predict = regressor_obj.predict(x_test)

#Visualize train set
plt.scatter(x_train, y_train, color = "red")
plt.plot(x_train, regressor_obj.predict(x_train), color = "blue")
plt.title("Salary Vs. Experience (For training data)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#Visualize test set
plt.scatter(x_test, y_test, color = "red")
#plt.plot(x_train, regressor_obj.predict(x_train), color = "blue")
plt.plot(x_test, regressor_obj.predict(x_test), color = "green")
plt.title("Salary Vs. Experience (For test data)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


#Question 1: How do I use my simple linear regression model to make a single prediction, #for example to predict the salary of an employee with 12 years of experience?

print(regressor_obj.predict([[12]]))

#Important note: Notice that the value of the feature (12 years) was input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting 12 into a double pair of square brackets makes the input exactly a 2D array. Simply put: 12→scalar, [12]→1D array ,[[12]]→2D array

#Question 2: How do I get the final regression equation y = b0 + b1 x with the final values of the coefficients b0 and b1?

print(regressor_obj.coef_.round(2))
print(regressor_obj.intercept_.round(2))

# Linear Equation for this model is, Salary = 26780.1 + 9312.58 * Experience