# -*- coding:utf-8 -*-
import numpy as np
from random import shuffle
# from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
      Structured SVM loss function, naive implementation (with loops).
    
      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.
    
      Inputs:
      - W: A numpy array of shape (D, C) containing weights.
      - X: A numpy array of shape (N, D) containing a minibatch of data.
      - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength
    
      Returns a tuple of:
      - loss as single float
      - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero
    
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)#1*num_classes
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            #Li=sum i!=yi max(0, Sj-Syi+delta)
            if margin > 0:
                loss += margin
                dW[:,j] += X[i].T
                dW[:,y[i]] += -X[i].T
    
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    
    # Add regularization to the loss.
#     the full Multiclass Support Vector Machine loss, 
#     which is made up of two components: 
#     the data loss (which is the average loss Li over all examples) 
#     and the regularization loss.
    loss += reg * np.sum(W * W)#W*W-->Each value in W is power two
    
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    dW += reg*W
    
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
      Structured SVM loss function, vectorized implementation.
    
      Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    
    # compute the loss and the gradient
    num_train = X.shape[0]
    num_classes = W.shape[1]
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    delta=1.0
    scores=X.dot(W)#-->num_train*num_classes
    # compute the margins for all classes in one vector operation
    # [range(num_train),list(y)] return the index of right label in each sample
    correct_class_score=scores[range(num_train),list(y)].reshape((-1,1))# num_train*1
    margins=np.maximum(0,scores-correct_class_score+delta)
    #max是比较一个数组（axis=0,1；列，行）
    #maximum是比较两个同size的数组
    margins[range(num_train),list(y)]=0
    loss=np.sum(margins)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    loss /= num_train
    loss += reg* np.sum(W*W)
    
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    coeff_mat = np.zeros((num_train, num_classes))
    coeff_mat[margins > 0] = 1
    coeff_mat[range(num_train), list(y)] = 0
    coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)
    
    dW = (X.T).dot(coeff_mat)#D * C
    dW = dW/num_train + reg*W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return loss, dW
