#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 20:54:18 2020

@author: hemalatha

Upper Confidence Bound
"""

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

#import data
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# UCB
num_of_users = 10000 #no. of users
num_of_advs = 10 # no. of advs
advs_selected = []
num_of_selections = [0] * num_of_advs
sum_of_rewards = [0] * num_of_advs
total_reward = 0

# for each user clicking the advs.
for each_user in range(0, num_of_users):
    ad = 0
    max_upper_bound = 0
    for each_adv in range(0, num_of_advs):
        # check if adv is selected, as it may result in infinity for initial click  
        if num_of_selections[each_adv] > 0:
            avg_reward = sum_of_rewards[each_adv] / num_of_selections[each_adv]
            delta_each_adv = math.sqrt(3/2 * math.log(each_user)/num_of_selections[each_adv])
            #calculate UCB
            upper_bound = avg_reward + delta_each_adv
        else:
            # set upper bound to infinity
            upper_bound = 1e400
        
        # determine max upper bound
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = each_adv
        
    advs_selected.append(ad)
    num_of_selections[ad] = num_of_selections[ad] + 1
    rewards = dataset.iloc[each_user, ad]
    sum_of_rewards[ad] = sum_of_rewards[ad] + rewards
    total_reward = total_reward + rewards
    
#Histograms
plt.hist(advs_selected)
plt.title("Histograms of advs selection")
plt.xlabel("Advs")
plt.ylabel("number of times each adv was selected")
plt.show()
            

