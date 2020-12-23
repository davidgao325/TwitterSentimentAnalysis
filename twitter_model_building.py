# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 19:19:22 2020

@author: david
"""
# Import the graphs and important classifiers
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
df = pd.read_csv("updated_tweet_info.csv")
data =  df.fillna(' ')
train,test = train_test_split(data, test_size = 0.2, random_state = 42)

# Categorize tweets into the training and testing data
train_clean_tweet=[]
for tweet in train['tweet']:
    train_clean_tweet.append(tweet)
test_clean_tweet=[]
for tweet in test['tweet']:
    test_clean_tweet.append(tweet)

# Convert Words into tokens and create classifiers
v = CountVectorizer(analyzer = "word")
train_features= v.fit_transform(train_clean_tweet)
test_features=v.transform(test_clean_tweet)
Classifiers = [ DecisionTreeClassifier(), BernoulliNB(alpha = 1), LogisticRegression(C = 6, max_iter = 1000, n_jobs=-1)]

# Convert features into arrays and then predict then fit the data
# Predict the accuracy of the data
# Create the confusion matrix
dense_features=train_features.toarray()
dense_test= test_features.toarray()
Accuracy=[]
Model=[]
for classifier in Classifiers:
    try:
        fit = classifier.fit(train_features,train['clean_polarity'])
        pred = fit.predict(test_features)
    except Exception:
        fit = classifier.fit(dense_features,train['clean_polarity'])
        pred = fit.predict(dense_test)
    accuracy = accuracy_score(pred,test['clean_polarity'])
    Accuracy.append(accuracy)
    Model.append(classifier.__class__.__name__)
    print(' Accuracy of '+classifier.__class__.__name__+' is '+str(accuracy))
    print(classification_report(pred,test['clean_polarity']))
    cm=confusion_matrix(pred , test['clean_polarity'])
    plt.figure()
    plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Reds)
    plt.xticks(range(2), ['Negative', 'Neutral', 'Positive'], fontsize=16,color='black')
    plt.yticks(range(2), ['Negative', 'Neutral', 'Positive'], fontsize=16)
    plt.show()

