#!/usr/bin/env python
#
# Naive Bayes Iris Classification
#
#  Copyright 2014 Tim O'Shea
# 
#  This classifier is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3, or (at your option)
#  any later version.
# 
#  This software is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
# sklearn iris example
#   http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
#
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation

# import some data to play with
iris = datasets.load_iris()
#clf = MultinomialNB(alpha=1)
clf = GaussianNB()

# Classifier ...
for NF in range(1,5):
    print iris.data[:,:NF].shape
    scores = cross_validation.cross_val_score( clf, iris.data[:,:NF], iris.target, cv=10)
    #scores = cross_validation.cross_val_score( clf, iris.data, iris.target, cv=10)
    print scores
    print("Accuracy: %0.2f (+/- %0.2f)  [Mis-classification = %0.2f]" % (scores.mean(), scores.std() * 2, 1-scores.mean()))

# Confusion Matrix ...
plt.figure(2, figsize=(8, 6))
plt.clf()
for i in range(0,4):
    for j in range(0,4):
        plt.subplot(4,4,4*i+j+1)
        plt.scatter(iris.data[:, i], iris.data[:, j], c=iris.target, cmap=plt.cm.Paired)
        plt.ylabel("Feature %d"%(i));
        plt.xlabel("Feature %d"%(j));
plt.show()
