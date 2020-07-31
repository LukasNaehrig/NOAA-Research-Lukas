# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 19:54:33 2020
@author: lukas Naehrig
Code Architecture: Lindsay Turner
Neural Network Model
Conversion from R
"""

##############################################################################
# IMPORT PACKAGES
##############################################################################

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense


import pandas as pd
import numpy as np
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.metrics import plot_confusion_matrix


# Takes a portion of samples from each class
# Note: this code is slow and needs revision
def undersample_ds(x, classCol, nsamples_class, seed):
    for i in np.unique(x[classCol]):
        if (sum(x[classCol] == i) - nsamples_class != 0):            
            xMatch = x[(x[classCol]).str.match(i)]
            x = x.drop(xMatch.sample(n = len(xMatch) - nsamples_class,
                                     random_state = seed).index)
    return x


##############################################################################
# IMPORT DATA & CREATE DATAFRAME
##############################################################################
    
dfAll = pd.read_csv(r'C:/Users/lnaeh/Desktop/NOAA Research/geo_data.csv')

nsamples_class = 10000 # Number of samples to take from each class
sample_seed = 42 # seed for random sample
#training_bc = undersample_ds(dfAll, 'Classname', nsamples_class, sample_seed)
training_bc = dfAll.groupby('Classname').apply(lambda s: s.sample(nsamples_class,
                                                                  random_state = sample_seed))

##############################################################################
# NEURAL NETWORK MODEL
##############################################################################
 
from sklearn.neural_network import MLPClassifier

X_train, X_test, y_train, y_test = train_test_split(features[feature_names],
                                                    labels, train_size = 0.9,
                                                    random_state = 42,
                                                    stratify = labels)
t0 = time.time()

# ???????????????????
nnet = clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                  hidden_layer_sizes=(15,), random_state=1)

nnet.fit(X_train, y_train)
t1 = time.time()
total_time = t1-t0                                                random_state = 42,
                                                    stratify = labels)
result = permutation_importance(nnet, X_train, y_train, random_state = 8)

predictions = nnet.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

confmat = confusion_matrix(y_test, predictions)
df_confmat = pd.DataFrame(confmat)
plot_confusion_matrix(nnet, X_test, y_test)
