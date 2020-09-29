#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:32:46 2020

@author: hemalatha
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import  pandas as pd

#Set working directory, Goto File explorer -> navigate to data path -> Save py file in same path -> run the file F5
dataset = pd.read_csv('Data.csv')
type(dataset)

# create independent variable X
X = dataset.iloc[:, :-1]

# create dependent variable Y
Y = dataset.iloc[:,3]

# Handle missing data
from sklearn.impute import SimpleImputer

# create an object of class Imputer (Note: Ctrl+I for inspecting command)
imputer = SimpleImputer(missing_values = np.nan , strategy="mean")
# fit imputer object to variable X, for 2nd and 3rd col and replace only those columns to be # imputed 
X.iloc[:,1:3] = imputer.fit_transform(X.iloc[:,1:3])

# Encoding categorical data or independent var
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
col_transformer_obj = ColumnTransformer(transformers = [("encoder" , OneHotEncoder(), [0] )], remainder = "passthrough")

X = np.array(col_transformer_obj.fit_transform(X))

# encoding dependent var
from sklearn.preprocessing import LabelEncoder
label_encode_obj = LabelEncoder()
Y = label_encode_obj.fit_transform(Y)

# Data splitting
from sklearn.model_selection import train_test_split
# random_state is used for seeding, tom ensure same random data is got in each run
x_train1, x_test1, y_train1, y_test1 = train_test_split(X, Y, test_size = 0.2, random_state = 1)

# Feature scaling
from sklearn.preprocessing import StandardScaler
std_scaler_obj = StandardScaler()
x_train1[:, 3:5] = std_scaler_obj.fit_transform(x_train1[:, 3:5])
x_test1[:, 3:5] = std_scaler_obj.transform(x_test1[:, 3:5])



