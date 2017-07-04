import numpy as np
from random import shuffle
# from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)
    
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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    num_train=X.shape[0]
    num_classes=W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
#       pass
    for i in range(num_train):
        scores = X[i].dot(W) #1*C
        scores -= np.max(scores)
        scores_exp_sum = np.sum(np.exp(scores))
        correct_class_score = scores[y[i]]
        loss += np.log(scores_exp_sum) - correct_class_score
        
        dW[:,y[i]] -= X[i]
        for j in range(num_classes):
            dW[:,j] += (np.exp(scores[j]) / scores_exp_sum) * X[i]
        
    loss = loss/num_train + 0.5*reg*np.sum(W*W)
    dW = dW/num_train + reg*W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
      Softmax loss function, vectorized version.
    
      Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    num_train=X.shape[0]
    num_classes=W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
#       pass
    socres= X.dot(W)#N * C
    socres -= np.max(socres, axis=1).reshape(-1,1)
    socres_exp_sum = np.sum(np.exp(socres) , axis=1).reshape(-1,1)
    socres_exp_norm = np.exp(socres) / socres_exp_sum
    loss = np.sum(- np.log(socres_exp_norm[range(num_train), list(y)]))
    loss = loss/num_train + 0.5*reg*np.sum(W*W)
    
    ds = socres_exp_norm.copy()
    ds[range(num_train), list(y)] -= 1
    dW = (X.T).dot(ds) #D * C
    dW = dW/num_train + reg*W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
    return loss, dW

