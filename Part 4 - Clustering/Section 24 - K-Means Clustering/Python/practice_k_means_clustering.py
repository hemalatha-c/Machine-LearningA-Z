#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 19:18:17 2020

@author: hemalatha

K-means clustering
"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import data
dataset = pd.read_csv('Mall_Customers.csv')
# Consider only 2 cols for to discover pattern with unsupervised learning
X = dataset.iloc[:, [3,4]].values

# using elbow method to determine optimal cluster number
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):# for 10 clusters range is 1,11
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
# plot elbow curve and determine no. of clusters
plt.plot(range(1,11), wcss)
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#train kmeans : optimal value from elbow curve = 5
kmeans_opt = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
# fit and predict the correct dependent var
y_pred = kmeans_opt.fit_predict(X)

# Visualization
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], c = "red", label = "Cluster1")
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c = "blue", label = "Cluster2")
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], c = "green", label = "Cluster3")
plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], c = "pink", label = "Cluster4")
plt.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], c = "orange", label = "Cluster5")
# plot centroids with kmeans obj
plt.scatter(kmeans_opt.cluster_centers_[:,0], kmeans_opt.cluster_centers_[:,1], c = 'yellow', label = "centroids")
plt.title("K-means clustering")
plt.xlabel('Annual income')
plt.ylabel('Spending score')
plt.show()