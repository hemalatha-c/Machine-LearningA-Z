#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:37:08 2020

@author: hemalatha

Thompson Sampling - To determine which advs has best conversion rate, in fewer iterations
"""

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# import data
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Thompson Sampling
num_of_users = 500 #10000
num_of_advs = 10
advs_selected = []

num_of_rewards_0 = [0] * num_of_advs
num_of_rewards_1 = [0] * num_of_advs

total_rewards = 0
# iterate through all users
for each_user in range(0, num_of_users):
    ad_selected = 0
    max_random = 0
    for each_adv in range(0, num_of_advs):
        random_draw_val = random.betavariate(num_of_rewards_1[each_adv]+1, num_of_rewards_0[each_adv] + 1)
        
        if random_draw_val > max_random:
            max_random = random_draw_val
            ad_selected = each_adv
    
    advs_selected.append(ad_selected)
    reward = dataset.iloc[each_user, ad_selected] 
    if reward == 1:
        num_of_rewards_1[ad_selected] += 1
    else:
        num_of_rewards_0[ad_selected] += 1
    
    total_rewards = total_rewards + reward

# Visualisation
plt.hist(advs_selected)
plt.title('Histogram of Advs using Thompson Sampling')
plt.xlabel('Advertisements')
plt.ylabel('Number of times advs were selected')
plt.show()

# Output Inference: Thompson Sampling works best in determining the advs with best conversion rate in less than 500 iterations