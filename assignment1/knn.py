# -*- coding:utf-8 -*-
'''
Created on 2017年5月27日

@author: samsung

The kNN classifier consists of two stages:
During training, the classifier takes the training data and simply remembers it
During testing, kNN classifies every test image by comparing to all training images 
and transfering the labels of the k most similar training examples

The value of k is cross-validated
In this exercise you will implement these steps and understand the basic 
Image Classification pipeline, cross-validation, and gain proficiency in writing efficient, 
vectorized code.
'''
import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# Load the raw CIFAR-10 data.
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape) 
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
# Training data shape:  (50000L, 32L, 32L, 3L)
# Training labels shape:  (50000L,)
# Test data shape:  (10000L, 32L, 32L, 3L)
# Test labels shape:  (10000L,)

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes): # index, value
    # a = np.array([1,0,1,1,0,0,1])
    # b = a==1 -->[ True False  True  True False False  True]
    # b = np.flatnonzero(a==1) --> [0 2 3 6]
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = range(num_training)
X_train = X_train[mask]#5000*32*32*3
y_train = y_train[mask]

num_test = 500
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))#5000*3072,确定行，列数自动定义
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

'''
We would now like to classify the test data with the kNN classifier. 

Recall that we can break down this process into two steps: 
First we must compute the distances between all test examples and all train examples. 
Given these distances, for each test example we find the k nearest examples 
and have them vote for the label

Lets begin with computing the distance matrix between all training and test examples. 
For example, if there are Ntr training examples and Nte test examples, 
this stage should result in a Nte x Ntr matrix where each element (i,j) 
is the distance between the i-th test and j-th train example.
'''

from cs231n.classifiers import KNearestNeighbor
# Create a kNN classifier instance. 
# Remember that training a kNN classifier is a noop: 
# the Classifier simply remembers the data and does no further processing 
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

# Open cs231n/classifiers/k_nearest_neighbor.py and implement
# compute_distances_two_loops.

# Test your implementation:
dists = classifier.compute_distances_two_loops(X_test)
print(dists.shape)
# (500L, 5000L)

# We can visualize the distance matrix: each row is a single test example and
# its distances to training examples
plt.imshow(dists, interpolation='none')
plt.show()

# Now implement the function predict_labels and run the code below:
# We use k = 1 (which is Nearest Neighbor).
y_test_pred = classifier.predict_labels(dists, k=1)

# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
# Got 137 / 500 correct => accuracy: 0.274000

y_test_pred = classifier.predict_labels(dists, k=5)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
# Got 139 / 500 correct => accuracy: 0.278000
# it's a slightly better performance than k=1

# Now lets speed up distance matrix computation by using partial vectorization
# with one loop. Implement the function compute_distances_one_loop and run the
# code below:
dists_one = classifier.compute_distances_one_loop(X_test)

'''
# To ensure that our vectorized implementation is correct, we make sure that it
# agrees with the naive implementation. 
# There are many ways to decide whether two matrices are similar; 
# one of the simplest is the Frobenius norm. In case you haven't seen it before, 
# the Frobenius norm of two matrices is the square root of the squared sum of differences of all elements; 
# in other words, reshape the matrices into vectors and compute the Euclidean distance between them.
'''
difference = np.linalg.norm(dists - dists_one, ord='fro')
print('Difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')
# Difference was: 0.000000
# Good! The distance matrices are the same

    
# Now implement the fully vectorized version inside compute_distances_no_loops
# and run the code
dists_two = classifier.compute_distances_no_loops(X_test)

# check that the distance matrix agrees with the one we computed before:
difference = np.linalg.norm(dists - dists_two, ord='fro')
print('Difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')
# Difference was: 0.000000
# Good! The distance matrices are the same
    
# Let's compare how fast the implementations are
def time_function(f, *args):
    """
    Call a function f with args and return the time (in seconds) that it took to execute.
        当函数的参数不确定时，可以使用*args 和**kwargs，*args 没有key值，**kwargs有key值
    *用来传递任意个无名字参数，这些参数会用一个tuple形式传递访问
    **用来传递任意个有名字的参数，这些参数用dict传递访问
    """
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic

two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
print('Two loop version took %f seconds' % two_loop_time)

one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
print('One loop version took %f seconds' % one_loop_time)

no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
print('No loop version took %f seconds' % no_loop_time)
# Two loop version took 37.813000 seconds
# One loop version took 81.474000 seconds
# No loop version took 0.432000 seconds

# you should see significantly faster performance with the fully vectorized implementation

'''
# Cross-validation
# We have implemented the k-Nearest Neighbor classifier but we set the value k = 5 arbitrarily.
# We will now determine the best value of this hyperparameter with cross-validation.
'''
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
# pass
y_train_=y_train.reshape((-1,1))
X_train_folds,y_train_folds=np.array_split(X_train, num_folds),np.array_split(y_train_, num_folds),
# x = np.arange(8.0)
# print(np.array_split(x, 3))
# [array([ 0.,  1.,  2.]), array([ 3.,  4.,  5.]), array([ 6.,  7.])]
# x[0] = [ 0.,  1.,  2.]
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}

################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
# pass
for k_ in k_choices:
    k_to_accuracies.setdefault(k_, [])
#     trainSet.setdefault(1,2) # 给字典trainSet 设置键为1 值为2的键值对 相当于trainSet[1]=2
#     print trainSet.setdefault(1,3)#如果已经存在1的键 那么setdefault的时候就不会改变他原有的值
#     trainSet[1]=3 #可以这样子改变它的1键值
for i in range(num_folds):
    classifier = KNearestNeighbor()
    X_val_train = np.vstack(X_train_folds[0:i] + X_train_folds[i+1:])#remove X_train_folds[i]
    y_val_train = np.vstack(y_train_folds[0:i] + y_train_folds[i+1:])
    y_val_train = y_val_train[:,0]
    classifier.train(X_val_train, y_val_train)
    for k_ in k_choices:
        y_val_pred = classifier.predict(X_train_folds[i], k=k_)
        num_correct = np.sum(y_val_pred == y_train_folds[i][:,0])
        accuracy = float(num_correct) / len(y_val_pred)
        k_to_accuracies[k_] = k_to_accuracies[k_] + [accuracy]
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))
# k = 1, accuracy = 0.263000
# k = 1, accuracy = 0.257000
# k = 1, accuracy = 0.264000
# k = 1, accuracy = 0.278000
# k = 1, accuracy = 0.266000
# k = 3, accuracy = 0.239000
# k = 3, accuracy = 0.249000
# k = 3, accuracy = 0.240000
# k = 3, accuracy = 0.266000
# k = 3, accuracy = 0.254000
# k = 5, accuracy = 0.248000
# k = 5, accuracy = 0.266000
# k = 5, accuracy = 0.280000
# k = 5, accuracy = 0.292000
# k = 5, accuracy = 0.280000
# k = 8, accuracy = 0.262000
# k = 8, accuracy = 0.282000
# k = 8, accuracy = 0.273000
# k = 8, accuracy = 0.290000
# k = 8, accuracy = 0.273000
# k = 10, accuracy = 0.265000
# k = 10, accuracy = 0.296000
# k = 10, accuracy = 0.276000
# k = 10, accuracy = 0.284000
# k = 10, accuracy = 0.280000
# k = 12, accuracy = 0.260000
# k = 12, accuracy = 0.295000
# k = 12, accuracy = 0.279000
# k = 12, accuracy = 0.283000
# k = 12, accuracy = 0.280000
# k = 15, accuracy = 0.252000
# k = 15, accuracy = 0.289000
# k = 15, accuracy = 0.278000
# k = 15, accuracy = 0.282000
# k = 15, accuracy = 0.274000
# k = 20, accuracy = 0.270000
# k = 20, accuracy = 0.279000
# k = 20, accuracy = 0.279000
# k = 20, accuracy = 0.282000
# k = 20, accuracy = 0.285000
# k = 50, accuracy = 0.271000
# k = 50, accuracy = 0.288000
# k = 50, accuracy = 0.278000
# k = 50, accuracy = 0.269000
# k = 50, accuracy = 0.266000
# k = 100, accuracy = 0.256000
# k = 100, accuracy = 0.270000
# k = 100, accuracy = 0.263000
# k = 100, accuracy = 0.256000
# k = 100, accuracy = 0.263000
       
# plot the raw observations
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()

# Based on the cross-validation results above, choose the best value for k,   
# retrain the classifier using all the training data, and test it on the test
# data. You should be able to get above 28% accuracy on the test data.
best_k = 10

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k=best_k)

# Compute and display the accuracy
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
# Got 141 / 500 correct => accuracy: 0.282000
