#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 14:07:09 2020

@author: hemalatha

Apriori algorithm :" Who bought this also bought that"
"""

#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import data
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

# Data preprocessing
transaction = []
for each_transac in range(0, len(dataset)):
    transaction.append([
        str(dataset.values[each_transac, each_val]) for each_val in range(0, 20)
        ])

# TRain apriori from apyori module
from apyori import apriori
rules = apriori(transactions = transaction, min_support = 0.003, min_confidence = 0.2,
                min_lift = 1, min_length = 2, max_length = 2)

rules_list =list(rules)

# to create dataframe of rules, to make it more readable
lhs = []
rhs = []
support_list = []
confidence_list = []
lift_list = []

for each_rule in rules_list:
    support_list.append(each_rule[1])
    confidence_list.append(each_rule[2][0][2])
    lift_list.append(each_rule[2][0][3])
    
    for each_obj in each_rule[2][0][0]:
       lhs.append(each_obj)
    for each_obj in each_rule[2][0][1]:
       rhs.append(each_obj)
       
df_list = list(zip(lhs,rhs, support_list, confidence_list, lift_list))

# non-sorted in order of relevenace/lift
df_res = pd.DataFrame(df_list, columns = ["left hand side", "right hand side", "Support", "Confidence", "Lift"])

#sorted in order of relevance
df_res_sorted = df_res.nlargest(n = 10, columns= 'Lift')