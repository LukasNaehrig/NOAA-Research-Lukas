# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:07:27 2020

@author: Lukas Naehrig
scikit classification

"""

import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'C:/Users/lnaeh/Desktop/NOAA Research/geo_data.csv')
df.head()
df.describe()

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

y = df['Classname']
x = df.drop(['Classname'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25, random_state=27)


# check from here on out:

clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, alpha=0.0001,
                     solver='sgd', verbose=10,  random_state=21, tol=0.0001)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)


accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cm

sns.heatmap(cm, center=True)
plt.show()

# https://www.kaggle.com/ahmethamzaemra/mlpclassifier-example

