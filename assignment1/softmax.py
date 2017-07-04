# -*- coding:utf-8 -*-
'''
Created on 2017.06.22

@author: samsung

Softmax exercise
This exercise is analogous to the SVM exercise. You will:
implement a fully-vectorized loss function for the Softmax classifier
implement the fully-vectorized expression for its analytic gradient
check your implementation with numerical gradient
use a validation set to tune the learning rate and regularization strength
optimize the loss function with SGD
visualize the final learned weights
'''

import numpy as np
import random
from cs231n.data_utils import get_CIFAR10_data
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# CIFAR-10 Data Loading and Preprocessing
#Load the row CIFAR-10 data
X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev=get_CIFAR10_data()
print(X_train.shape,X_val.shape,X_test.shape,X_dev.shape)
print(y_train.shape,y_val.shape,y_test.shape,y_dev.shape)

# ((49000L, 3073L), (1000L, 3073L), (1000L, 3073L), (500L, 3073L))
# ((49000L,), (1000L,), (1000L,), (500L,))

'''Softmax Classifier'''
# First implement the naive softmax loss function with nested loops.
# Open the file cs231n/classifiers/softmax.py and implement the
# softmax_loss_naive function.

from cs231n.classifiers.softmax import softmax_loss_naive
from cs231n.classifiers.softmax import softmax_loss_vectorized
import time
# Generate a random softmax weight matrix and use it to compute the loss.
W = np.random.randn(3073, 10) * 0.0001
loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)
# loss, grad = softmax_loss_vectorized(W, X_dev, y_dev, 0.0)

# As a rough sanity check, our loss should be something close to -log(0.1).
print('loss: %f' % loss)
print('sanity check: %f' % (-np.log(0.1)))
# loss: 2.323639
# sanity check: 2.302585

# Why do we expect our loss to be close to -log(0.1)? Explain briefly.**
# Your answer: Since the weight matrix W is uniform randomly selected, 
# the predicted probability of each class is uniform distribution 
# and identically equals 1/10, where 10 is the number of classes. 
# So the cross entroy for each example is -log(0.1), 
# which should equal to the loss.

# Complete the implementation of softmax_loss_naive and implement a (naive)
# version of the gradient that uses nested loops.
# loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)

# As we did for the SVM, use numeric gradient checking as a debugging tool.
# The numeric gradient should be close to the analytic gradient.

from cs231n.gradient_check import grad_check_sparse
f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad, 10)
 
# similar to SVM case, do another gradient check with regularization
loss, grad = softmax_loss_naive(W, X_dev, y_dev, 1e2)
f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 1e2)[0]
grad_numerical = grad_check_sparse(f, W, grad, 10)

# numerical: -1.359503 analytic: -1.359503, relative error: 1.988460e-08
# numerical: -1.644227 analytic: -1.644227, relative error: 3.427574e-08
# numerical: -0.004066 analytic: -0.004066, relative error: 2.066353e-05
# numerical: -0.838385 analytic: -0.838386, relative error: 8.709942e-08
# numerical: -1.087934 analytic: -1.087934, relative error: 1.259007e-08
# numerical: 1.186072 analytic: 1.186072, relative error: 9.508286e-09
# numerical: 0.095718 analytic: 0.095718, relative error: 1.630505e-07
# numerical: -1.598150 analytic: -1.598150, relative error: 5.764799e-08
# numerical: -1.105070 analytic: -1.105071, relative error: 1.024094e-07
# numerical: 1.451380 analytic: 1.451380, relative error: 2.636540e-08
# numerical: 1.986223 analytic: 1.986223, relative error: 4.082614e-08
# numerical: 0.040157 analytic: 0.040157, relative error: 2.393237e-06
# numerical: 4.423382 analytic: 4.423382, relative error: 1.400176e-08
# numerical: 1.876353 analytic: 1.876353, relative error: 3.025306e-08
# numerical: 4.393903 analytic: 4.393903, relative error: 1.727682e-08
# numerical: 1.891919 analytic: 1.891919, relative error: 1.773744e-08
# numerical: -1.041932 analytic: -1.041932, relative error: 4.072054e-08
# numerical: -0.827540 analytic: -0.827540, relative error: 6.683947e-08
# numerical: -0.367617 analytic: -0.367617, relative error: 4.193331e-08
# numerical: -1.021289 analytic: -1.021289, relative error: 9.526995e-09

# Now that we have a naive implementation of the softmax loss function and its gradient,
# implement a vectorized version in softmax_loss_vectorized.
# The two versions should compute the same results, but the vectorized version should be
# much faster.
reg=0.00001
 
tic = time.time()
loss_naive, grad_naive = softmax_loss_naive(W, X_dev, y_dev, reg)
toc = time.time()
print('naive loss: %e computed in %f s' % (loss_naive, toc - tic))
 
tic = time.time()
loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_dev, y_dev, reg)
toc = time.time()
print('vectorized loss: %e computed in %f s' % (loss_vectorized, toc - tic))
 
# As we did for the SVM, we use the Frobenius norm to compare the two versions
# of the gradient.
grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print('Loss difference: %f' % np.abs(loss_naive-loss_vectorized))
print('Gradient difference: %f' % grad_difference)

# naive loss: 2.323639e+00 computed in 0.132000 s
# vectorized loss: 2.323639e+00 computed in 0.366000 s
# Loss difference: 0.000000
# Gradient difference: 0.000000

# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of over 0.35 on the validation set.

from cs231n.classifiers import Softmax
results = {}
best_val = -1
best_softmax = None
learning_rates = [1e-7, 2e-7, 5e-7]
#regularization_strengths = [5e4, 1e8]
regularization_strengths =[(1+0.1*i)*1e4 for i in range(-3,4)] + [(5+0.1*i)*1e4 for i in range(-3,4)]

################################################################################
# TODO:                                                                        #
# Use the validation set to set the learning rate and regularization strength. #
# This should be identical to the validation that you did for the SVM; save    #
# the best trained softmax classifer in best_softmax.                          #
################################################################################
for lr in learning_rates:
    for rs in regularization_strengths:
        softmax = Softmax()
        softmax.train(X_train, y_train, lr, rs, num_iters=2000)
        y_train_pred = softmax.predict(X_train)
        train_accuracy = np.mean(y_train == y_train_pred)
        
        y_val_pred = softmax.predict(X_val)
        val_accuracy = np.mean(y_val == y_val_pred)
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_softmax = softmax
        results[(lr,rs)] = train_accuracy, val_accuracy
    
#print out results
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %r reg %e train accuracy: %f val accuracy: %f' % (
        lr, reg, train_accuracy, val_accuracy))
print('best validation accuracy achieved during cross-validation: %f' % best_val)

# lr 1e-07 reg 7.000000e+03 train accuracy: 0.342204 val accuracy: 0.351000
# lr 1e-07 reg 8.000000e+03 train accuracy: 0.346510 val accuracy: 0.351000
# lr 1e-07 reg 9.000000e+03 train accuracy: 0.354837 val accuracy: 0.369000
# lr 1e-07 reg 1.000000e+04 train accuracy: 0.358633 val accuracy: 0.385000
# lr 1e-07 reg 1.100000e+04 train accuracy: 0.361041 val accuracy: 0.359000
# lr 1e-07 reg 1.200000e+04 train accuracy: 0.359388 val accuracy: 0.374000
# lr 1e-07 reg 1.300000e+04 train accuracy: 0.363224 val accuracy: 0.362000
# lr 1e-07 reg 4.700000e+04 train accuracy: 0.329898 val accuracy: 0.341000
# lr 1e-07 reg 4.800000e+04 train accuracy: 0.329694 val accuracy: 0.340000
# lr 1e-07 reg 4.900000e+04 train accuracy: 0.328959 val accuracy: 0.346000
# lr 1e-07 reg 5.000000e+04 train accuracy: 0.324612 val accuracy: 0.342000
# lr 1e-07 reg 5.100000e+04 train accuracy: 0.327122 val accuracy: 0.340000
# lr 1e-07 reg 5.200000e+04 train accuracy: 0.326694 val accuracy: 0.340000
# lr 1e-07 reg 5.300000e+04 train accuracy: 0.325531 val accuracy: 0.338000
# lr 2e-07 reg 7.000000e+03 train accuracy: 0.379776 val accuracy: 0.383000
# lr 2e-07 reg 8.000000e+03 train accuracy: 0.378102 val accuracy: 0.385000
# lr 2e-07 reg 9.000000e+03 train accuracy: 0.376041 val accuracy: 0.392000
# lr 2e-07 reg 1.000000e+04 train accuracy: 0.376061 val accuracy: 0.391000
# lr 2e-07 reg 1.100000e+04 train accuracy: 0.371592 val accuracy: 0.386000
# lr 2e-07 reg 1.200000e+04 train accuracy: 0.368898 val accuracy: 0.382000
# lr 2e-07 reg 1.300000e+04 train accuracy: 0.363449 val accuracy: 0.377000
# lr 2e-07 reg 4.700000e+04 train accuracy: 0.331143 val accuracy: 0.338000
# lr 2e-07 reg 4.800000e+04 train accuracy: 0.333286 val accuracy: 0.348000
# lr 2e-07 reg 4.900000e+04 train accuracy: 0.323878 val accuracy: 0.338000
# lr 2e-07 reg 5.000000e+04 train accuracy: 0.327857 val accuracy: 0.345000
# lr 2e-07 reg 5.100000e+04 train accuracy: 0.326796 val accuracy: 0.347000
# lr 2e-07 reg 5.200000e+04 train accuracy: 0.321735 val accuracy: 0.340000
# lr 2e-07 reg 5.300000e+04 train accuracy: 0.329224 val accuracy: 0.342000
# lr 5e-07 reg 7.000000e+03 train accuracy: 0.379898 val accuracy: 0.393000
# lr 5e-07 reg 8.000000e+03 train accuracy: 0.379286 val accuracy: 0.391000
# lr 5e-07 reg 9.000000e+03 train accuracy: 0.373857 val accuracy: 0.385000
# lr 5e-07 reg 1.000000e+04 train accuracy: 0.373184 val accuracy: 0.378000
# lr 5e-07 reg 1.100000e+04 train accuracy: 0.363367 val accuracy: 0.385000
# lr 5e-07 reg 1.200000e+04 train accuracy: 0.368796 val accuracy: 0.388000
# lr 5e-07 reg 1.300000e+04 train accuracy: 0.359245 val accuracy: 0.377000
# lr 5e-07 reg 4.700000e+04 train accuracy: 0.325184 val accuracy: 0.340000
# lr 5e-07 reg 4.800000e+04 train accuracy: 0.327735 val accuracy: 0.340000
# lr 5e-07 reg 4.900000e+04 train accuracy: 0.324551 val accuracy: 0.340000
# lr 5e-07 reg 5.000000e+04 train accuracy: 0.330184 val accuracy: 0.337000
# lr 5e-07 reg 5.100000e+04 train accuracy: 0.319653 val accuracy: 0.329000
# lr 5e-07 reg 5.200000e+04 train accuracy: 0.331020 val accuracy: 0.347000
# lr 5e-07 reg 5.300000e+04 train accuracy: 0.330327 val accuracy: 0.347000

# evaluate on test set
# Evaluate the best softmax on test set
y_test_pred = best_softmax.predict(X_test)
test_accuracy = np.mean(y_test_pred == y_test)
print('softmax on raw pixels final test set accuracy: %f' % (test_accuracy, ))
# best validation accuracy achieved during cross-validation: 0.393000
# softmax on raw pixels final test set accuracy: 0.372000


# Visualize the learned weights for each class
w = best_softmax.W[:-1,:]
w = w.reshape(32,32,3,10)

w_min, w_max = np.min(w), np.max(w)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(len(classes)):
    plt.subplot(2, len(classes)/2, i+1)
    # Rescale the weights to be between 0 and 255
    wimg = 255.0 * (w[:,:,:,i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype('uint8'))
    plt.title(classes[i])
plt.show()
