#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 16:06:07 2020

@author: hemalatha

hierarchical clustering
"""

#Import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import data
dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:,[3,4]].values

# Find optimal no. of clusters using dendrogram in scipy
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidiean Distance')
plt.show()

# Either 3 or 5 clusters is optimal

#Train HC model
from sklearn.cluster import AgglomerativeClustering
hc_obj = AgglomerativeClustering(n_clusters=5) # 3 or 5 clusters
y_pred = hc_obj.fit_predict(X)


# visualize HC for 3 clusters
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], c = 'red', label='Cluster-1')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c = 'blue', label='Cluster-2')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], c = 'green', label='Cluster-3')
plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], c = 'magenta', label='Cluster-4')
plt.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], c = 'cyan', label='Cluster-5')
plt.title('Hierarchical Clustering')
plt.xlabel('Annual INcome')
plt.ylabel('Spending score')
plt.show()