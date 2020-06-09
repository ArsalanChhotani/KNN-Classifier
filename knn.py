# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 15:48:56 2019

@author: Arsalan Ashraf
"""



import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
#from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
names = ['sepsal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv("E:\\DS\\iris.data", names=names)
print(dataset)

dataset.shape

print(dataset.groupby('class').size())

array = dataset.values
X = array[:,0:4]

Y = array[:,4]
t_size = 0.20
seed = 6
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=t_size, random_state=seed)
print(Y_test)


knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))

