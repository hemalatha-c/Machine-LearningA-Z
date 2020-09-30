#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 14:22:39 2020

@author: hemalatha

NLP- Sentiment Analysis for restaurant reviews

"""

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import data
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter= '\t', quoting= 3)

#cleaning of text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
corpus = []

for each_review_index in range(0,1000):
    # replace all non-char with space in each review
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][each_review_index])
    #transform caps to small letters
    review = review.lower()
    # split by words
    review = review.split()
    #Stemming
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(each_word) for each_word in review if not each_word in set(all_stopwords)]
    # convert list to string
    review = ' '.join(review)
    corpus.append(review)

#build bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

#split data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#build a classifier for sentiment classification
from sklearn.naive_bayes import GaussianNB
classifier_naive_base = GaussianNB()
classifier_naive_base.fit(x_train, y_train)

#predict test data
y_pred = classifier_naive_base.predict(x_test)
y_result = np.concatenate((np.reshape(y_test, (len(y_test),1)), np.reshape(y_pred, (len(y_pred),1))),1)

#confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
conf_matrix = confusion_matrix(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred)

#Practice homework
#single review challenge
user_review = "i liked Desserts over Starters"
user_review = re.sub('[^a-zA-Z]', ' ', user_review).lower().split()
user_review = [ps.stem(each_word) for each_word in user_review if not each_word in set(all_stopwords)]
user_review = " ".join(user_review)
print(user_review)
user_review_vec = cv.fit_transform([user_review]).toarray()
print(user_review_vec)
#y_pred_single_review = classifier_naive_base.predict(np.reshape(user_review_vec, 3, 1566))