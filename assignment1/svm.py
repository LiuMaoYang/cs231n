# -*- coding:utf-8 -*-
'''
Created on 2017年6月13日

@author: samsung

 Multiclass Support Vector Machine exercise
 In this exercise you will:
 implement a fully-vectorized loss function for the SVM
 implement the fully-vectorized expression for its analytic gradient
 check your implementation using numerical gradient
 use a validation set to tune the learning rate and regularization strength
 optimize the loss function with SGD
 visualize the final learned weights
'''
import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

# This is a bit of magic to make matplotlib figures appear inline in the
# notebook rather than in a new window.
# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# %load_ext autoreload
# %autoreload 2

''' 
CIFAR-10 Data Loading and Preprocessing：
Split the data into train, val, and test sets.

Preprocessing: subtract the mean image
 first: compute the image mean based on the training data
 second: subtract the mean image from train and test data
 third: append the bias dimension of ones (i.e. bias trick) so that our SVM
'''
#Load the row CIFAR-10 data
cifa10_dir='cs231n/datasets/cifar-10-batches-py'
X_train,y_train,X_test,y_test=load_CIFAR10(cifa10_dir)
# As a sanity check, we print out the size of the training and test data.

print('Training data shape',X_train.shape)
print('Training labels shape',y_train.shape)
print('Testing data shape',X_test.shape)
print('Testing labels shape',y_test.shape)
# ('Training data shape', (50000L, 32L, 32L, 3L))
# ('Training labels shape', (50000L,))
# ('Testing data shape', (10000L, 32L, 32L, 3L))
# ('Testing labels shape', (10000L,))

#Visualize some examples from the dataset.
#We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_class=len(classes)
samples_per_class=7
for y,cls in enumerate(classes):
    #Return the indices of the non-zero elements of the input array
    idxs=np.flatnonzero(y_train==y)
    #Generate a uniform random sample from idexs of size samples_per_class without replacement(无放回)
    idexs=np.random.choice(idxs,samples_per_class,replace=False)
    for i,idx in enumerate(idexs):
        plt_idx=i*num_class+y+1
        #samples_per_class row,num_class column, No.plt_idx subgraph
        plt.subplot(samples_per_class,num_class,plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i==0:
            plt.title(cls)
plt.show()

# Split the data into train, val, and test sets. In addition we will
# create a small development set as a subset of the training data;
# we can use this for development so our code runs faster.
num_training=49000
num_validation=1000
num_test=1000
num_dev=500

# Our validation set will be num_validation points from the original
# training set.
mask=range(num_training,num_training+num_validation)
X_val=X_train[mask]
y_val=y_train[mask]

# Our training set will be the first num_train points from the original
# training set.
mask=range(num_training)
X_train=X_train[mask]
y_train=y_train[mask]

#We also make a development set, which is a small subset of the training set
mask=np.random.choice(num_training,num_dev,replace=False)
X_dev=X_train[mask]
y_dev=y_train[mask]

#We use the first num_test points of the original test set as our test set
mask=range(num_test)
X_test=X_test[mask]
y_test=y_test[mask]

print('Train data shape',X_train.shape)
print('Train labels shape',y_train.shape)
print('Validation data shape',X_val.shape)
print('Validation labels shape',y_val.shape)
print('Test data shape',X_test.shape)
print('Test labels shape',y_test.shape)
 
# Train data shape:  (49000L, 32L, 32L, 3L)
# Train labels shape:  (49000L,)
# Validation data shape:  (1000L, 32L, 32L, 3L)
# Validation labels shape:  (1000L,)
# Test data shape:  (1000L, 32L, 32L, 3L)
# Test labels shape:  (1000L,)

#Preprocessing: reshape the image data into rows
X_train=np.reshape(X_train, (num_training,-1))
X_val=np.reshape(X_val, (num_validation,-1))
X_test=np.reshape(X_test, (num_test,-1))
X_dev=np.reshape(X_dev, (num_dev,-1))

#As a sanity check, print out the shapes of the data
print('Training data shape',X_train.shape)
print('Validation data shape',X_val.shape)
print('Testing data shape',X_test.shape)
print('Development data shape',X_dev.shape)
# Training data shape:  (49000L, 3072L)
# Validation data shape:  (1000L, 3072L)
# Test data shape:  (1000L, 3072L)
# dev data shape:  (500L, 3072L)

#Preprocessing: subtract the mean image
#first:compute the image mean based on the training data
#Note: only form the training data, not form the entire data
mean_image=np.mean(X_train,axis=0)#(3072L,),这里reshape(1,-1)与否并无所谓
print(mean_image[:10])#print a few of elements
plt.figure(figsize=(4,4))
plt.imshow(mean_image.reshape((32,32,3)).astype('uint8'))#visualization the mean image
plt.show()

#second:substract the mean image from train and test data
X_train-=mean_image
X_val-=mean_image
X_test-=mean_image
X_dev-=mean_image

#third: append the bias dimension of ones (i.e. bias trick) 
#so that our SVM only has to worry about optimizing a single weight matrix W
#for a sample: x=[x1,x2,...,xn], we transport it into y=[x1,x2,...,xn,1]
X_train=np.hstack([X_train,np.ones((num_training,1))])
# or X_train=np.concatenate((X_train,np.ones((num_training,1))),axis=1)
X_val=np.hstack([X_val,np.ones((num_validation,1))])
X_test=np.hstack([X_test,np.ones((num_test,1))])
X_dev=np.hstack([X_dev,np.ones((num_dev,1))])

print(X_train.shape,X_val.shape,X_test.shape,X_dev.shape)
# (49000L, 3073L) (1000L, 3073L) (1000L, 3073L) (500L, 3073L)


'''SVM Classifier'''
# Evaluate the naive implementation of the loss we provided for you:
from cs231n.classifiers.linear_svm import svm_loss_naive
import time

# generate a random SVM weight matrix of small numbers
W = np.random.randn(3073, 10) * 0.0001 

loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.00001)
print('loss: %f' % (loss, ))
# loss: 8.680625
 
# Once you've implemented the gradient, recompute it with the code below
# and gradient check it with the function we provided for you
 
# Compute the loss and its gradient at W.
loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.0)
 
# Numerically compute the gradient along several randomly chosen dimensions, and
# compare them with your analytically computed gradient. The numbers should match
# almost exactly along all dimensions.
from cs231n.gradient_check import grad_check_sparse
f = lambda w: svm_loss_naive(w, X_dev, y_dev, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad)
 
# do the gradient check once again with regularization turned on
# you didn't forget the regularization gradient did you?
loss, grad = svm_loss_naive(W, X_dev, y_dev, 1e2)
f = lambda w: svm_loss_naive(w, X_dev, y_dev, 1e2)[0]
grad_numerical = grad_check_sparse(f, W, grad)

# numerical: 4.750041 analytic: 2375.020653, relative error: 9.960080e-01
# numerical: -0.613890 analytic: -272.153061, relative error: 9.954988e-01
# numerical: 5.204069 analytic: 2602.034531, relative error: 9.960080e-01
# numerical: -9.965576 analytic: -4982.788082, relative error: 9.960080e-01
# numerical: -2.208331 analytic: -1104.165388, relative error: 9.960080e-01
# numerical: -0.205572 analytic: -102.786102, relative error: 9.960080e-01
# numerical: 20.532350 analytic: 10228.862776, relative error: 9.959935e-01
# numerical: 25.898624 analytic: 12949.312082, relative error: 9.960080e-01
# numerical: -32.864109 analytic: -16432.054571, relative error: 9.960080e-01
# numerical: -13.374846 analytic: -6693.496531, relative error: 9.960116e-01
# numerical: 5.350977 analytic: 2632.819113, relative error: 9.959434e-01
# numerical: -0.647188 analytic: -331.811162, relative error: 9.961067e-01
# numerical: -55.241853 analytic: -27622.450719, relative error: 9.960082e-01
# numerical: -21.379517 analytic: -10681.285025, relative error: 9.960048e-01
# numerical: -17.861294 analytic: -8932.428054, relative error: 9.960088e-01
# numerical: -48.852475 analytic: -24437.255216, relative error: 9.960098e-01
# numerical: 10.901939 analytic: 5454.631110, relative error: 9.960107e-01
# numerical: 0.303558 analytic: 174.044605, relative error: 9.965178e-01
# numerical: 20.949192 analytic: 10476.408063, relative error: 9.960087e-01
# numerical: 14.822717 analytic: 7412.337143, relative error: 9.960085e-01

# Next implement the function svm_loss_vectorized; for now only compute the loss;
# we will implement the gradient in a moment.
tic = time.time()
loss_naive, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.00001)
toc = time.time()
print ('Naive loss: %e computed in %fs' % (loss_naive, toc - tic))
 
from cs231n.classifiers.linear_svm import svm_loss_vectorized
tic = time.time()
loss_vectorized, _ = svm_loss_vectorized(W, X_dev, y_dev, 0.00001)
toc = time.time()
print ('Vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))
 
# The losses should match but your vectorized implementation should be much faster.
print ('difference: %f' % (loss_naive - loss_vectorized))

# Naive loss: 8.832338e+00 computed in 1.064000s
# Vectorized loss: 8.832338e+00 computed in 0.549000s
# difference: -0.000000

'''
Stochastic Gradient Descent
We now have vectorized and efficient expressions for the loss, 
the gradient and our gradient matches the numerical gradient. 
We are therefore ready to do SGD to minimize the loss.
'''

# In the file linear_classifier.py, implement SGD in the function
# LinearClassifier.train() and then run it with the code below.
from cs231n.classifiers import LinearSVM
svm = LinearSVM()
tic = time.time()
loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=5e4,
                      num_iters=1500, verbose=True)
toc = time.time()
print ('That took %fs' % (toc - tic))

# iteration 0 / 1500: loss 1559.246212
# iteration 100 / 1500: loss 569.509740
# iteration 200 / 1500: loss 210.736050
# iteration 300 / 1500: loss 80.617356
# iteration 400 / 1500: loss 32.712619
# iteration 500 / 1500: loss 15.715612
# iteration 600 / 1500: loss 9.447349
# iteration 700 / 1500: loss 6.621553
# iteration 800 / 1500: loss 6.557098
# iteration 900 / 1500: loss 6.009901
# iteration 1000 / 1500: loss 5.900652
# iteration 1100 / 1500: loss 5.751258
# iteration 1200 / 1500: loss 6.241348
# iteration 1300 / 1500: loss 5.494885
# iteration 1400 / 1500: loss 5.716264
# That took 12.395000s

# A useful debugging strategy is to plot the loss as a function of
# iteration number:
plt.plot(loss_hist)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()

# Write the LinearSVM.predict function and evaluate the performance on both the
# training and validation set
y_train_pred = svm.predict(X_train)
print ('training accuracy: %f' % (np.mean(y_train == y_train_pred), ))
y_val_pred = svm.predict(X_val)
print ('validation accuracy: %f' % (np.mean(y_val == y_val_pred), ))
# training accuracy: 0.368061
# validation accuracy: 0.373000

# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of about 0.4 on the validation set.
learning_rates = [1.4e-7, 1.5e-7, 1.6e-7]
regularization_strengths = [(1+i*0.1)*1e4 for i in range(-3,3)] + [(2+0.1*i)*1e4 for i in range(-3,3)]

# results is dictionary mapping tuples of the form
# (learning_rate, regularization_strength) to tuples of the form
# (training_accuracy, validation_accuracy). The accuracy is simply the fraction
# of data points that are correctly classified.
results = {}
best_val = -1   # The highest validation accuracy that we have seen so far.
best_svm = None # The LinearSVM object that achieved the highest validation rate.

################################################################################
# TODO:                                                                        #
# Write code that chooses the best hyperparameters by tuning on the validation #
# set. For each combination of hyperparameters, train a linear SVM on the      #
# training set, compute its accuracy on the training and validation sets, and  #
# store these numbers in the results dictionary. In addition, store the best   #
# validation accuracy in best_val and the LinearSVM object that achieves this  #
# accuracy in best_svm.                                                        #
#                                                                              #
# Hint: You should use a small value for num_iters as you develop your         #
# validation code so that the SVMs don't take much time to train; once you are #
# confident that your validation code works, you should rerun the validation   #
# code with a larger value for num_iters.                                      #
################################################################################
for rs in regularization_strengths:
    for lr in learning_rates:
        svm = LinearSVM()
        loss_hist = svm.train(X_train, y_train, lr, rs, num_iters=3000)
        
        y_train_pred = svm.predict(X_train)
        train_accuracy = np.mean(y_train == y_train_pred)
        y_val_pred = svm.predict(X_val)
        val_accuracy = np.mean(y_val == y_val_pred)
        
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_svm = svm           
        results[(lr,rs)] = train_accuracy, val_accuracy
#pass
################################################################################
#                              END OF YOUR CODE                                #
################################################################################
    
# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print ('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy))
    
print ('best validation accuracy achieved during cross-validation: %f' % best_val)

# lr 1.400000e-07 reg 7.000000e+03 train accuracy: 0.391469 val accuracy: 0.388000
# lr 1.400000e-07 reg 8.000000e+03 train accuracy: 0.394367 val accuracy: 0.389000
# lr 1.400000e-07 reg 9.000000e+03 train accuracy: 0.394694 val accuracy: 0.396000
# lr 1.400000e-07 reg 1.000000e+04 train accuracy: 0.390388 val accuracy: 0.398000
# lr 1.400000e-07 reg 1.100000e+04 train accuracy: 0.393184 val accuracy: 0.389000
# lr 1.400000e-07 reg 1.200000e+04 train accuracy: 0.391224 val accuracy: 0.391000
# lr 1.400000e-07 reg 1.700000e+04 train accuracy: 0.382673 val accuracy: 0.403000
# lr 1.400000e-07 reg 1.800000e+04 train accuracy: 0.387041 val accuracy: 0.386000
# lr 1.400000e-07 reg 1.900000e+04 train accuracy: 0.385918 val accuracy: 0.392000
# lr 1.400000e-07 reg 2.000000e+04 train accuracy: 0.381939 val accuracy: 0.376000
# lr 1.400000e-07 reg 2.100000e+04 train accuracy: 0.383612 val accuracy: 0.373000
# lr 1.400000e-07 reg 2.200000e+04 train accuracy: 0.378286 val accuracy: 0.384000
# lr 1.500000e-07 reg 7.000000e+03 train accuracy: 0.398898 val accuracy: 0.393000
# lr 1.500000e-07 reg 8.000000e+03 train accuracy: 0.391061 val accuracy: 0.388000
# lr 1.500000e-07 reg 9.000000e+03 train accuracy: 0.391000 val accuracy: 0.386000
# lr 1.500000e-07 reg 1.000000e+04 train accuracy: 0.388286 val accuracy: 0.387000
# lr 1.500000e-07 reg 1.100000e+04 train accuracy: 0.389020 val accuracy: 0.409000
# lr 1.500000e-07 reg 1.200000e+04 train accuracy: 0.383551 val accuracy: 0.391000
# lr 1.500000e-07 reg 1.700000e+04 train accuracy: 0.384224 val accuracy: 0.383000
# lr 1.500000e-07 reg 1.800000e+04 train accuracy: 0.387429 val accuracy: 0.384000
# lr 1.500000e-07 reg 1.900000e+04 train accuracy: 0.380429 val accuracy: 0.373000
# lr 1.500000e-07 reg 2.000000e+04 train accuracy: 0.384245 val accuracy: 0.403000
# lr 1.500000e-07 reg 2.100000e+04 train accuracy: 0.380449 val accuracy: 0.382000
# lr 1.500000e-07 reg 2.200000e+04 train accuracy: 0.380878 val accuracy: 0.390000
# lr 1.600000e-07 reg 7.000000e+03 train accuracy: 0.399429 val accuracy: 0.397000
# lr 1.600000e-07 reg 8.000000e+03 train accuracy: 0.393633 val accuracy: 0.398000
# lr 1.600000e-07 reg 9.000000e+03 train accuracy: 0.389837 val accuracy: 0.386000
# lr 1.600000e-07 reg 1.000000e+04 train accuracy: 0.392265 val accuracy: 0.408000
# lr 1.600000e-07 reg 1.100000e+04 train accuracy: 0.391592 val accuracy: 0.391000
# lr 1.600000e-07 reg 1.200000e+04 train accuracy: 0.389306 val accuracy: 0.377000
# lr 1.600000e-07 reg 1.700000e+04 train accuracy: 0.381163 val accuracy: 0.378000
# lr 1.600000e-07 reg 1.800000e+04 train accuracy: 0.379551 val accuracy: 0.368000
# lr 1.600000e-07 reg 1.900000e+04 train accuracy: 0.377755 val accuracy: 0.376000
# lr 1.600000e-07 reg 2.000000e+04 train accuracy: 0.373959 val accuracy: 0.388000
# lr 1.600000e-07 reg 2.100000e+04 train accuracy: 0.380204 val accuracy: 0.382000
# lr 1.600000e-07 reg 2.200000e+04 train accuracy: 0.378653 val accuracy: 0.393000
# best validation accuracy achieved during cross-validation: 0.409000

# Visualize the cross-validation results
import math
x_scatter = [math.log10(x[0]) for x in results]
y_scatter = [math.log10(x[1]) for x in results]

# plot training accuracy
marker_size = 100
colors = [results[x][0] for x in results]
plt.subplot(2, 1, 1)
plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 training accuracy')

# plot validation accuracy
colors = [results[x][1] for x in results] # default size of markers is 20
plt.subplot(2, 1, 2)
plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 validation accuracy')
plt.show()

# Evaluate the best svm on test set
y_test_pred = best_svm.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)
print ('linear SVM on raw pixels final test set accuracy: %f' % test_accuracy)
# linear SVM on raw pixels final test set accuracy: 0.372000

# Visualize the learned weights for each class.
# Depending on your choice of learning rate and regularization strength, these may
# or may not be nice to look at.
w = best_svm.W[:-1,:] # strip out the bias, 行从0至倒数第二行，列取:
w = w.reshape(32, 32, 3, 10)
w_min, w_max = np.min(w), np.max(w)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in xrange(10):
    plt.subplot(2, 5, i + 1)
        
    # Rescale the weights to be between 0 and 255
    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])
plt.show()