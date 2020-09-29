#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 16:42:05 2020

@author: hemalatha

K- Nearest Neighbour
"""
# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import data
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y , test_size = 0.2, random_state = 0)

# Train model
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(x_train, y_train)

# Prediction confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
print('Confusion matrix:\n', confusion_matrix(y_test, knn_classifier.predict(x_test)))
print('Accuracy:\n', accuracy_score(y_test, knn_classifier.predict(x_test)))

# After Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)

knn_classifier_sc = KNeighborsClassifier()
knn_classifier_sc.fit(x_train_sc, y_train)
y_pred = knn_classifier_sc.predict(x_test_sc)
print('\n\nAfter Feature scaling\nConfusion matrix:\n', confusion_matrix(y_test, y_pred))
print('Accuracy:\n', accuracy_score(y_test,y_pred))


from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
plt.contourf(X1, X2, knn_classifier_sc.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()