#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:08:45 2020

@author: hemalatha

Eclat Learning
"""

#Import libraries
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

# TRain apriori from apyori module [Same for eclat as it uses only Support feature]
from apyori import apriori
rules = apriori(transactions = transaction, min_support = 0.003, min_confidence = 0.2,
                min_lift = 1, min_length = 2, max_length = 2)

rules_list =list(rules)

# to create dataframe of rules, to make it more readable
prod_1 = []
prod_2 = []
support_list = []

for each_rule in rules_list:
    support_list.append(each_rule[1])
    
    for each_obj in each_rule[2][0][0]:
       prod_1.append(each_obj)
    for each_obj in each_rule[2][0][1]:
       prod_2.append(each_obj)
       
df_list = list(zip(prod_1, prod_2, support_list))

# non-sorted in order of relevenace/lift
df_res = pd.DataFrame(df_list, columns = ["Product-1", "Product-2", "Support"])

#sorted in order of relevance
df_res_sorted = df_res.sort_values('Support')