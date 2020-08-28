# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 11:23:32 2020
@author: lukas Naehrig
sklearn test

"""

### Package import:
from sklearn.neural_network import MLPClassifier

### Training:
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)

### Predict:
clf.predict([[2., 2.], [-1., -2.]])

[coef.shape for coef in clf.coefs_]

clf.predict_proba([[2., 2.], [1., 2.]])






# multi-label classification:
#X = [[0., 0.], [1., 1.]]
#y = [[0, 1], [1, 1]]
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                     hidden_layer_sizes=(15,), random_state=1)
#clf.fit(X, y)
#clf.predict([[1., 2.]])
#clf.predict([[0., 0.]])



# Notes:
# X: training data
# y: class names
