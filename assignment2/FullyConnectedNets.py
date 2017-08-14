#encoding: utf-8
'''
date:2017.07.04

Fully-Connected Neural Nets
In the previous homework you implemented a fully-connected two-layer neural network on CIFAR-10. 
The implementation was simple but not very modular since the loss and gradient were computed in a single monolithic function. 
This is manageable for a simple two-layer network, but would become impractical as we move to bigger models. 
Ideally we want to build networks using a more modular design so that we can implement different layer types in isolation 
and then snap them together into models with different architectures.

In this exercise we will implement fully-connected networks using a more modular approach. 
For each layer we will implement a forward and a backward function. 
The forward function will receive inputs, weights, and other parameters and will return both an output and 
a cache object storing data needed for the backward pass, like this:

def layer_forward(x, w):
  """ Receive inputs x and weights w """
  # Do some computations ...
  z = # ... some intermediate value
  # Do some more computations ...
  out = # the output

  cache = (x, w, z, out) # Values we need to compute gradients

  return out, cache
  
The backward pass will receive upstream derivatives and the cache object, 
and will return gradients with respect to the inputs and weights, like this:

def layer_backward(dout, cache):
  """
  Receive derivative of loss with respect to outputs and cache,
  and compute derivative with respect to inputs.
  """
  # Unpack cache values
  x, w, z, out = cache

  # Use values in cache to compute derivatives
  dx = # Derivative of loss with respect to x
  dw = # Derivative of loss with respect to w

  return dx, dw
  
After implementing a bunch of layers this way, we will be able to easily combine them to build classifiers with different architectures.
In addition to implementing fully-connected networks of arbitrary depth, we will also explore different update rules 
for optimization, and introduce Dropout as a regularizer and Batch Normalization as a tool to more efficiently optimize deep networks.
'''

import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def rel_error(x, y):
    ''' returns relative error'''
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# # Load the (preprocessed) CIFAR10 data.
data = get_CIFAR10_data()
for k, v in data.iteritems():
    print '%s: ' % k, v.shape
 
# X_val:  (1000L, 3L, 32L, 32L)
# X_train:  (49000L, 3L, 32L, 32L)
# X_test:  (1000L, 3L, 32L, 32L)
# y_val:  (1000L,)
# y_train:  (49000L,)
# y_test:  (1000L,)

'''Affine layer: foward'''
# Test the affine_forward function

num_inputs = 2
input_shape = (4, 5, 6)
output_dim = 3
   
# np.prod: Return the product of array elements over a given axis.
  
input_size = num_inputs * np.prod(input_shape) # 2 * 4 * 5 * 6
weight_size = output_dim * np.prod(input_shape)
 
x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape) # (N, d_1, ..., d_k) (2, 4, 5, 6)
w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim) #(D, M) (4 * 5 * 6, 3)
b = np.linspace(-0.3, 0.1, num=output_dim) #(M, )
 
out, _ = affine_forward(x, w, b)
correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
                        [ 3.25553199,  3.5141327,   3.77273342]])
 
# Compare your output with ours. The error should be around 1e-9.
print 'Testing affine_forward function:'
print 'difference: ', rel_error(out, correct_out)
# Testing affine_forward function:
# difference:  9.76984946819e-10

'''Affine layer: backward'''
# Test the affine_backward function

x = np.random.randn(10, 2, 3)
w = np.random.randn(6, 5)
b = np.random.randn(5)
dout = np.random.randn(10, 5)
 
dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout)
 
_, cache = affine_forward(x, w, b)
dx, dw, db = affine_backward(dout, cache)
 
# The error should be around 1e-10
print 'Testing affine_backward function:'
print 'dx error: ', rel_error(dx_num, dx)
print 'dw error: ', rel_error(dw_num, dw)
print 'db error: ', rel_error(db_num, db)
# Testing affine_backward function:
# dx error:  2.2345671641e-10
# dw error:  3.79557892455e-11
# db error:  1.24055368425e-11

'''ReLU layer: forward'''
# Test the relu_forward function

x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)
out, _ = relu_forward(x)
correct_out = np.array([[ 0.,          0.,          0.,          0.,        ],
                        [ 0.,          0.,          0.04545455,  0.13636364,],
                        [ 0.22727273,  0.31818182,  0.40909091,  0.5,       ]])
 
# Compare your output with ours. The error should be around 1e-8
print 'Testing relu_forward function:'
print 'difference: ', rel_error(out, correct_out)
# Testing relu_forward function:
# difference:  4.99999979802e-08

'''"Sandwich" layers'''
# There are some common patterns of layers that are frequently used in neural nets. 
# For example, affine layers are frequently followed by a ReLU nonlinearity. 
 
from cs231n.layer_utils import affine_relu_forward, affine_relu_backward
 
x = np.random.randn(2, 3, 4)
w = np.random.randn(12, 10)
b = np.random.randn(10)
dout = np.random.randn(2, 10)
 
out, cache = affine_relu_forward(x, w, b)
dx, dw, db = affine_relu_backward(dout, cache)
 
dx_num = eval_numerical_gradient_array(lambda x: affine_relu_forward(x, w, b)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: affine_relu_forward(x, w, b)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: affine_relu_forward(x, w, b)[0], b, dout)

print 'Testing affine_relu_forward:'
print 'dx error: ', rel_error(dx_num, dx)
print 'dw error: ', rel_error(dw_num, dw)
print 'db error: ', rel_error(db_num, db)
# Testing affine_relu_forward:
# dx error:  8.92879185539e-11
# dw error:  4.41889147037e-10
# db error:  7.82670260359e-12

'''Loss layers: Softmax and SVM'''
num_classes, num_inputs = 10, 50
x = 0.001 * np.random.randn(num_inputs, num_classes)
y = np.random.randint(num_classes, size=num_inputs)
 
dx_num = eval_numerical_gradient(lambda x: svm_loss(x, y)[0], x, verbose=False)
loss, dx = svm_loss(x, y)
 
# Test svm_loss function. Loss should be around 9 and dx error should be 1e-9
print 'Testing svm_loss:'
print 'loss: ', loss
print 'dx error: ', rel_error(dx_num, dx)
 
dx_num = eval_numerical_gradient(lambda x: softmax_loss(x, y)[0], x, verbose=False)
loss, dx = softmax_loss(x, y)
 
# Test softmax_loss function. Loss should be 2.3 and dx error should be 1e-8
print '\nTesting softmax_loss:'
print 'loss: ', loss
print 'dx error: ', rel_error(dx_num, dx)
# Testing svm_loss:
# loss:  8.99900450642
# dx error:  3.0387355051e-09
# 
# Testing softmax_loss:
# loss:  2.30248602163
# dx error:  9.49036194391e-09

'''
Two-layer network
The architecure should be affine - relu - affine - softmax.
'''
N, D, H, C = 3, 5, 50, 7
X = np.random.randn(N, D)
y = np.random.randint(C, size=N)
  
std = 1e-2
model = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=std)
  
print 'Testing initialization ... '
W1_std = abs(model.params['W1'].std() - std)
b1 = model.params['b1']
W2_std = abs(model.params['W2'].std() - std)
b2 = model.params['b2']
# assert是用来检查一个条件，如果它为真，就不做任何事。
# 如果它为假，则会抛出AssertError并且包含错误信息
assert W1_std < std / 10, 'First layer weights do not seem right'
assert np.all(b1 == 0), 'First layer biases do not seem right'
assert W2_std < std / 10, 'Second layer weights do not seem right'
assert np.all(b2 == 0), 'Second layer biases do not seem right'
  
print 'Testing test-time forward pass ... '
model.params['W1'] = np.linspace(-0.7, 0.3, num=D*H).reshape(D, H)
model.params['b1'] = np.linspace(-0.1, 0.9, num=H)
model.params['W2'] = np.linspace(-0.3, 0.4, num=H*C).reshape(H, C)
model.params['b2'] = np.linspace(-0.9, 0.1, num=C)
X = np.linspace(-5.5, 4.5, num=N*D).reshape(D, N).T
scores = model.loss(X)
correct_scores = np.asarray(
  [[11.53165108,  12.2917344,   13.05181771,  13.81190102,  14.57198434, 15.33206765,  16.09215096],
   [12.05769098,  12.74614105,  13.43459113,  14.1230412,   14.81149128, 15.49994135,  16.18839143],
   [12.58373087,  13.20054771,  13.81736455,  14.43418138,  15.05099822, 15.66781506,  16.2846319 ]])
scores_diff = np.abs(scores - correct_scores).sum()
assert scores_diff < 1e-6, 'Problem with test-time forward pass'
  
print 'Testing training loss (no regularization)'
y = np.asarray([0, 5, 1])
loss, grads = model.loss(X, y)
correct_loss = 3.4702243556
assert abs(loss - correct_loss) < 1e-10, 'Problem with training-time loss'
  
model.reg = 1.0
loss, grads = model.loss(X, y)
correct_loss = 26.5948426952
assert abs(loss - correct_loss) < 1e-10, 'Problem with regularization loss'
  
for reg in [0.0, 0.7]:
    print 'Running numeric gradient check with reg = ', reg
    model.reg = reg
    loss, grads = model.loss(X, y)
      
    for name in sorted(grads):
        f = lambda _: model.loss(X, y)[0]
        grad_num = eval_numerical_gradient(f, model.params[name], verbose=False)
        print '%s relative error: %.2e' % (name, rel_error(grad_num, grads[name]))
 
# Testing initialization ... 
# Testing test-time forward pass ... 
# Testing training loss (no regularization)
# Running numeric gradient check with reg =  0.0
# W1 relative error: 1.83e-08
# W2 relative error: 3.12e-10
# b1 relative error: 9.83e-09
# b2 relative error: 4.33e-10
# Running numeric gradient check with reg =  0.7
# W1 relative error: 2.53e-07
# W2 relative error: 2.85e-08
# b1 relative error: 1.56e-08
# b2 relative error: 7.76e-10

'''Solver'''
model = TwoLayerNet(reg=1e-1)
solver = None
##############################################################################
# TODO: Use a Solver instance to train a TwoLayerNet that achieves at least  #
# 50% accuracy on the validation set.                                        #
##############################################################################
solver = Solver(model, data,
    update_rule='sgd',
    optim_config={'learning_rate': 1e-3,},
    lr_decay=0.8,
    num_epochs=10, batch_size=100,
    print_every=100)
solver.train()
scores = model.loss(data['X_test'])
y_pred = np.argmax(scores, axis = 1)
acc = np.mean(y_pred == data['y_test'])
print 'test acc: %f' %(acc)

# (Iteration 1 / 4900) loss: 2.323448
# (Epoch 0 / 10) train acc: 0.113000; val_acc: 0.110000
# (Iteration 101 / 4900) loss: 1.843075
# (Iteration 201 / 4900) loss: 1.589917
# (Iteration 301 / 4900) loss: 1.632272
# (Iteration 401 / 4900) loss: 1.686838
# (Epoch 1 / 10) train acc: 0.468000; val_acc: 0.439000
# (Iteration 501 / 4900) loss: 1.540372
# (Iteration 601 / 4900) loss: 1.781505
# (Iteration 701 / 4900) loss: 1.475611
# (Iteration 801 / 4900) loss: 1.474844
# (Iteration 901 / 4900) loss: 1.419930
# (Epoch 2 / 10) train acc: 0.511000; val_acc: 0.472000
# (Iteration 1001 / 4900) loss: 1.437217
# (Iteration 1101 / 4900) loss: 1.366095
# (Iteration 1201 / 4900) loss: 1.314775
# (Iteration 1301 / 4900) loss: 1.385588
# (Iteration 1401 / 4900) loss: 1.351293
# (Epoch 3 / 10) train acc: 0.526000; val_acc: 0.488000
# (Iteration 1501 / 4900) loss: 1.475149
# (Iteration 1601 / 4900) loss: 1.182712
# (Iteration 1701 / 4900) loss: 1.368307
# (Iteration 1801 / 4900) loss: 1.430352
# (Iteration 1901 / 4900) loss: 1.424367
# (Epoch 4 / 10) train acc: 0.503000; val_acc: 0.497000
# (Iteration 2001 / 4900) loss: 1.242272
# (Iteration 2101 / 4900) loss: 1.149194
# (Iteration 2201 / 4900) loss: 1.168347
# (Iteration 2301 / 4900) loss: 1.294638
# (Iteration 2401 / 4900) loss: 1.178249
# (Epoch 5 / 10) train acc: 0.545000; val_acc: 0.495000
# (Iteration 2501 / 4900) loss: 1.202747
# (Iteration 2601 / 4900) loss: 1.376193
# (Iteration 2701 / 4900) loss: 1.128453
# (Iteration 2801 / 4900) loss: 1.109438
# (Iteration 2901 / 4900) loss: 1.158221
# (Epoch 6 / 10) train acc: 0.570000; val_acc: 0.506000
# (Iteration 3001 / 4900) loss: 1.101283
# (Iteration 3101 / 4900) loss: 1.193702
# (Iteration 3201 / 4900) loss: 1.170483
# (Iteration 3301 / 4900) loss: 1.343353
# (Iteration 3401 / 4900) loss: 1.356165
# (Epoch 7 / 10) train acc: 0.614000; val_acc: 0.506000
# (Iteration 3501 / 4900) loss: 1.088773
# (Iteration 3601 / 4900) loss: 1.111282
# (Iteration 3701 / 4900) loss: 1.168451
# (Iteration 3801 / 4900) loss: 1.453259
# (Iteration 3901 / 4900) loss: 1.256485
# (Epoch 8 / 10) train acc: 0.588000; val_acc: 0.518000
# (Iteration 4001 / 4900) loss: 1.262684
# (Iteration 4101 / 4900) loss: 1.038075
# (Iteration 4201 / 4900) loss: 1.080302
# (Iteration 4301 / 4900) loss: 1.176514
# (Iteration 4401 / 4900) loss: 1.068464
# (Epoch 9 / 10) train acc: 0.632000; val_acc: 0.521000
# (Iteration 4501 / 4900) loss: 1.229950
# (Iteration 4601 / 4900) loss: 1.169083
# (Iteration 4701 / 4900) loss: 1.210292
# (Iteration 4801 / 4900) loss: 1.002819
# (Epoch 10 / 10) train acc: 0.631000; val_acc: 0.531000
# test acc: 0.527000

# Run this cell to visualize training loss and train / val accuracy

plt.subplot(211), plt.title('Traing loss')
plt.plot(solver.loss_history, 'o'), plt.xlabel('Iteration')
 
plt.subplot(212), plt.title('Accuracy')
plt.plot(solver.train_acc_history, 'o', label = 'train')
plt.plot(solver.val_acc_history, 'o', label = 'val')
plt.plot([0.5] * len(solver.val_acc_history), 'k--') # 在0.5处画虚线
plt.xlabel('Epoch')
plt.legend(loc = 'lower right')
plt.gcf().set_size_inches(15, 12)
plt.show()

'''Multilayer network'''

'''Initial loss and gradient check'''
N, D, H1, H2, C = 2, 15, 20, 30, 10
X = np.random.randn(N, D)
y = np.random.randint(C, size = (N, ))
 
for reg in [0, 3.14]:
    print 'Running check with reg = ', reg
    model = FullyConnectedNet((H1, H2), input_dim=D, num_classes=C, 
                              reg=reg, weight_scale=5e-2, dtype=np.float64)
    loss, grads = model.loss(X, y)
    print 'Initial loss: ', loss
     
    for name in sorted(grads):
        f = lambda _: model.loss(X, y)[0]
        grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
        print '%s relative error: %.2e' % (name, rel_error(grad_num, grads[name]))
        
# Running check with reg =  0
# Initial loss:  2.30563828618
# W1 relative error: 6.89e-07
# W2 relative error: 2.02e-07
# W3 relative error: 3.78e-07
# b1 relative error: 1.47e-08
# b2 relative error: 6.08e-09
# b3 relative error: 8.47e-11
# 
# Running check with reg =  3.14
# Initial loss:  7.32472003351
# W1 relative error: 1.06e-07
# W2 relative error: 4.30e-07
# W3 relative error: 4.22e-08
# b1 relative error: 4.29e-08
# b2 relative error: 4.42e-09
# b3 relative error: 2.99e-10

# TODO: Use a three-layer Net to overfit 50 training examples.
num_train = 50
small_data = {
    'X_train': data['X_train'][:num_train],
    'y_train': data['y_train'][:num_train],
    'X_val': data['X_val'],
    'y_val': data['y_val'],
    }
weight_scale = 1e-2
learning_rate = 8e-3
model = FullyConnectedNet([100, 100],
              weight_scale=weight_scale, dtype=np.float64)
solver = Solver(model, small_data,
                print_every=10, num_epochs=20, batch_size=25,
                update_rule='sgd',
                optim_config={
                  'learning_rate': learning_rate,
                }
         )
solver.train()
plt.plot(solver.loss_history, 'o')
plt.title('Training loss history')
plt.xlabel('Iteration')
plt.ylabel('Training loss')
plt.show()

# (Iteration 1 / 40) loss: 2.326388
# (Epoch 0 / 20) train acc: 0.300000; val_acc: 0.113000
# (Epoch 1 / 20) train acc: 0.280000; val_acc: 0.166000
# (Epoch 2 / 20) train acc: 0.360000; val_acc: 0.175000
# (Epoch 3 / 20) train acc: 0.440000; val_acc: 0.193000
# (Epoch 4 / 20) train acc: 0.540000; val_acc: 0.179000
# (Epoch 5 / 20) train acc: 0.540000; val_acc: 0.166000
# (Iteration 11 / 40) loss: 0.992183
# (Epoch 6 / 20) train acc: 0.600000; val_acc: 0.176000
# (Epoch 7 / 20) train acc: 0.740000; val_acc: 0.206000
# (Epoch 8 / 20) train acc: 0.780000; val_acc: 0.201000
# (Epoch 9 / 20) train acc: 0.900000; val_acc: 0.208000
# (Epoch 10 / 20) train acc: 0.880000; val_acc: 0.204000
# (Iteration 21 / 40) loss: 0.434243
# (Epoch 11 / 20) train acc: 0.840000; val_acc: 0.189000
# (Epoch 12 / 20) train acc: 0.960000; val_acc: 0.210000
# (Epoch 13 / 20) train acc: 0.960000; val_acc: 0.208000
# (Epoch 14 / 20) train acc: 0.980000; val_acc: 0.223000
# (Epoch 15 / 20) train acc: 0.960000; val_acc: 0.222000
# (Iteration 31 / 40) loss: 0.121148
# (Epoch 16 / 20) train acc: 0.980000; val_acc: 0.215000
# (Epoch 17 / 20) train acc: 1.000000; val_acc: 0.210000
# (Epoch 18 / 20) train acc: 0.980000; val_acc: 0.207000
# (Epoch 19 / 20) train acc: 1.000000; val_acc: 0.198000
# (Epoch 20 / 20) train acc: 0.980000; val_acc: 0.214000

# TODO: Use a five-layer Net to overfit 50 training examples.

num_train = 50
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}
 
learning_rate = 3e-4
weight_scale = 1e-1 #1e-5
model = FullyConnectedNet([100, 100, 100, 100],
                weight_scale=weight_scale, dtype=np.float64)
solver = Solver(model, small_data,
                print_every=10, num_epochs=20, batch_size=25,
                update_rule='sgd',
                optim_config={
                  'learning_rate': learning_rate,
                }
         )
solver.train()
 
plt.plot(solver.loss_history, 'o')
plt.title('Training loss history')
plt.xlabel('Iteration')
plt.ylabel('Training loss')
plt.show()

# (Iteration 1 / 40) loss: 115.629093
# (Epoch 0 / 20) train acc: 0.100000; val_acc: 0.072000
# (Epoch 1 / 20) train acc: 0.260000; val_acc: 0.096000
# (Epoch 2 / 20) train acc: 0.300000; val_acc: 0.116000
# (Epoch 3 / 20) train acc: 0.420000; val_acc: 0.122000
# (Epoch 4 / 20) train acc: 0.660000; val_acc: 0.110000
# (Epoch 5 / 20) train acc: 0.680000; val_acc: 0.109000
# (Iteration 11 / 40) loss: 8.341693
# (Epoch 6 / 20) train acc: 0.700000; val_acc: 0.117000
# (Epoch 7 / 20) train acc: 0.860000; val_acc: 0.109000
# (Epoch 8 / 20) train acc: 0.940000; val_acc: 0.111000
# (Epoch 9 / 20) train acc: 0.920000; val_acc: 0.112000
# (Epoch 10 / 20) train acc: 0.860000; val_acc: 0.112000
# (Iteration 21 / 40) loss: 0.791873
# (Epoch 11 / 20) train acc: 0.800000; val_acc: 0.118000
# (Epoch 12 / 20) train acc: 0.840000; val_acc: 0.119000
# (Epoch 13 / 20) train acc: 0.900000; val_acc: 0.117000
# (Epoch 14 / 20) train acc: 0.960000; val_acc: 0.119000
# (Epoch 15 / 20) train acc: 0.960000; val_acc: 0.119000
# (Iteration 31 / 40) loss: 2.310050
# (Epoch 16 / 20) train acc: 0.940000; val_acc: 0.110000
# (Epoch 17 / 20) train acc: 0.940000; val_acc: 0.112000
# (Epoch 18 / 20) train acc: 0.960000; val_acc: 0.118000
# (Epoch 19 / 20) train acc: 1.000000; val_acc: 0.122000
# (Epoch 20 / 20) train acc: 1.000000; val_acc: 0.121000

'''Update rules'''

'''
SGD+Momentum
Stochastic gradient descent with momentum is a widely used update rule 
that tends to make deep networks converge faster than vanilla stochstic gradient descent.
'''
from cs231n.optim import sgd_momentum
N, D = 4, 5
w = np.linspace(-0.4, 0.6, num=N*D).reshape(N, D)
dw = np.linspace(-0.6, 0.4, num=N*D).reshape(N, D)
v = np.linspace(0.6, 0.9, num=N*D).reshape(N, D)
config = {'learning_rate': 1e-3, 'velocity': v}
 
next_w, _ = sgd_momentum(w, dw, config)
 
expected_next_w = np.asarray([
  [ 0.1406,      0.20738947,  0.27417895,  0.34096842,  0.40775789],
  [ 0.47454737,  0.54133684,  0.60812632,  0.67491579,  0.74170526],
  [ 0.80849474,  0.87528421,  0.94207368,  1.00886316,  1.07565263],
  [ 1.14244211,  1.20923158,  1.27602105,  1.34281053,  1.4096    ]])
expected_velocity = np.asarray([
  [ 0.5406,      0.55475789,  0.56891579, 0.58307368,  0.59723158],
  [ 0.61138947,  0.62554737,  0.63970526,  0.65386316,  0.66802105],
  [ 0.68217895,  0.69633684,  0.71049474,  0.72465263,  0.73881053],
  [ 0.75296842,  0.76712632,  0.78128421,  0.79544211,  0.8096    ]])
 
print 'next_w error: ', rel_error(next_w, expected_next_w)
print 'velocity error: ', rel_error(expected_velocity, config['velocity'])

# next_w error:  8.88234703351e-09
# velocity error:  4.26928774328e-09

num_train = 4000
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}
  
solvers = {}
  
for update_rule in ['sgd', 'sgd_momentum']:
    print 'running_with', update_rule
    model = FullyConnectedNet([100, 100, 100, 100, 100], weight_scale=5e-2)
  
    solver = Solver(model, small_data,
                  num_epochs=5, batch_size=100,
                  update_rule=update_rule,
                  optim_config={
                    'learning_rate': 1e-2,
                  },
                  verbose=False)
    solvers[update_rule] = solver
    solver.train()
    print
 
plt.subplot(3, 1, 1)
plt.title('Training loss')
plt.xlabel('Iteration')
 
plt.subplot(3, 1, 2)
plt.title('Training accuracy')
plt.xlabel('Epoch')
 
plt.subplot(3, 1, 3)
plt.title('Validation accuracy')
plt.xlabel('Epoch')
 
for update_rule, solver in solvers.iteritems():
    plt.subplot(3, 1, 1)
    plt.plot(solver.loss_history, 'o', label=update_rule)
       
    plt.subplot(3, 1, 2)
    plt.plot(solver.train_acc_history, '-o', label=update_rule)
     
    plt.subplot(3, 1, 3)
    plt.plot(solver.val_acc_history, '-o', label=update_rule)
   
for i in [1, 2, 3]:
    plt.subplot(3, 1, i)
    plt.legend(loc='upper center', ncol=4)
     
plt.gcf().set_size_inches(15, 15)
plt.show()

# running_with sgd
# (Iteration 1 / 200) loss: 2.704285
# (Epoch 0 / 5) train acc: 0.109000; val_acc: 0.093000
# (Iteration 11 / 200) loss: 2.286599
# (Iteration 21 / 200) loss: 2.229084
# (Iteration 31 / 200) loss: 2.179731
# (Epoch 1 / 5) train acc: 0.284000; val_acc: 0.227000
# (Iteration 41 / 200) loss: 1.960889
# (Iteration 51 / 200) loss: 1.980615
# (Iteration 61 / 200) loss: 2.063655
# (Iteration 71 / 200) loss: 1.905524
# (Epoch 2 / 5) train acc: 0.322000; val_acc: 0.286000
# (Iteration 81 / 200) loss: 1.923180
# (Iteration 91 / 200) loss: 1.906682
# (Iteration 101 / 200) loss: 1.843918
# (Iteration 111 / 200) loss: 1.863513
# (Epoch 3 / 5) train acc: 0.390000; val_acc: 0.306000
# (Iteration 121 / 200) loss: 1.880490
# (Iteration 131 / 200) loss: 1.793785
# (Iteration 141 / 200) loss: 1.778961
# (Iteration 151 / 200) loss: 1.629362
# (Epoch 4 / 5) train acc: 0.422000; val_acc: 0.317000
# (Iteration 161 / 200) loss: 1.825020
# (Iteration 171 / 200) loss: 1.639313
# (Iteration 181 / 200) loss: 1.563712
# (Iteration 191 / 200) loss: 1.678494
# (Epoch 5 / 5) train acc: 0.430000; val_acc: 0.347000
# 
# running_with sgd_momentum
# (Iteration 1 / 200) loss: 2.836213
# (Epoch 0 / 5) train acc: 0.115000; val_acc: 0.137000
# (Iteration 11 / 200) loss: 2.119565
# (Iteration 21 / 200) loss: 1.967207
# (Iteration 31 / 200) loss: 1.655524
# (Epoch 1 / 5) train acc: 0.283000; val_acc: 0.306000
# (Iteration 41 / 200) loss: 1.963470
# (Iteration 51 / 200) loss: 1.745253
# (Iteration 61 / 200) loss: 1.710365
# (Iteration 71 / 200) loss: 1.803677
# (Epoch 2 / 5) train acc: 0.418000; val_acc: 0.365000
# (Iteration 81 / 200) loss: 1.715369
# (Iteration 91 / 200) loss: 1.506558
# (Iteration 101 / 200) loss: 1.784505
# (Iteration 111 / 200) loss: 1.689894
# (Epoch 3 / 5) train acc: 0.430000; val_acc: 0.338000
# (Iteration 121 / 200) loss: 1.657494
# (Iteration 131 / 200) loss: 1.523051
# (Iteration 141 / 200) loss: 1.591500
# (Iteration 151 / 200) loss: 1.483424
# (Epoch 4 / 5) train acc: 0.449000; val_acc: 0.330000
# (Iteration 161 / 200) loss: 1.577505
# (Iteration 171 / 200) loss: 1.440392
# (Iteration 181 / 200) loss: 1.280600
# (Iteration 191 / 200) loss: 1.512771
# (Epoch 5 / 5) train acc: 0.482000; val_acc: 0.317000

'''
RMSProp and Adam
RMSProp and Adam are update rules that set per-parameter learning rates 
by using a running average of the second moments of gradients.
'''
# Test RMSProp implementation; you should see errors less than 1e-7

from cs231n.optim import rmsprop
 
N, D = 4, 5
w = np.linspace(-0.4, 0.6, num=N*D).reshape(N, D)
dw = np.linspace(-0.6, 0.4, num=N*D).reshape(N, D)
cache = np.linspace(0.6, 0.9, num=N*D).reshape(N, D)
 
config = {'learning_rate': 1e-2, 'cache': cache}
next_w, _ = rmsprop(w, dw, config=config)
 
expected_next_w = np.asarray([
  [-0.39223849, -0.34037513, -0.28849239, -0.23659121, -0.18467247],
  [-0.132737,   -0.08078555, -0.02881884,  0.02316247,  0.07515774],
  [ 0.12716641,  0.17918792,  0.23122175,  0.28326742,  0.33532447],
  [ 0.38739248,  0.43947102,  0.49155973,  0.54365823,  0.59576619]])
expected_cache = np.asarray([
  [ 0.5976,      0.6126277,   0.6277108,   0.64284931,  0.65804321],
  [ 0.67329252,  0.68859723,  0.70395734,  0.71937285,  0.73484377],
  [ 0.75037008,  0.7659518,   0.78158892,  0.79728144,  0.81302936],
  [ 0.82883269,  0.84469141,  0.86060554,  0.87657507,  0.8926    ]])
 
print 'next_w error: ', rel_error(expected_next_w, next_w)
print 'cache error: ', rel_error(expected_cache, config['cache'])
# 
# next_w error:  9.52468751104e-08
# cache error:  2.64779558072e-09

# Test Adam implementation; you should see errors around 1e-7 or less

from cs231n.optim import adam
 
N, D = 4, 5
w = np.linspace(-0.4, 0.6, num=N*D).reshape(N, D)
dw = np.linspace(-0.6, 0.4, num=N*D).reshape(N, D)
m = np.linspace(0.6, 0.9, num=N*D).reshape(N, D)
v = np.linspace(0.7, 0.5, num=N*D).reshape(N, D)
 
config = {'learning_rate': 1e-2, 'm': m, 'v': v, 't': 5}
next_w, _ = adam(w, dw, config=config)
 
expected_next_w = np.asarray([
  [-0.40094747, -0.34836187, -0.29577703, -0.24319299, -0.19060977],
  [-0.1380274,  -0.08544591, -0.03286534,  0.01971428,  0.0722929],
  [ 0.1248705,   0.17744702,  0.23002243,  0.28259667,  0.33516969],
  [ 0.38774145,  0.44031188,  0.49288093,  0.54544852,  0.59801459]])
expected_v = np.asarray([
  [ 0.69966,     0.68908382,  0.67851319,  0.66794809,  0.65738853,],
  [ 0.64683452,  0.63628604,  0.6257431,   0.61520571,  0.60467385,],
  [ 0.59414753,  0.58362676,  0.57311152,  0.56260183,  0.55209767,],
  [ 0.54159906,  0.53110598,  0.52061845,  0.51013645,  0.49966,   ]])
expected_m = np.asarray([
  [ 0.48,        0.49947368,  0.51894737,  0.53842105,  0.55789474],
  [ 0.57736842,  0.59684211,  0.61631579,  0.63578947,  0.65526316],
  [ 0.67473684,  0.69421053,  0.71368421,  0.73315789,  0.75263158],
  [ 0.77210526,  0.79157895,  0.81105263,  0.83052632,  0.85      ]])
 
print 'next_w error: ', rel_error(expected_next_w, next_w)
print 'v error: ', rel_error(expected_v, config['v'])
print 'm error: ', rel_error(expected_m, config['m'])
# 
# next_w error:  1.13956917985e-07
# v error:  4.20831403811e-09
# m error:  4.21496319311e-09
# 
learning_rates = {'rmsprop': 1e-4, 'adam': 1e-3}
for update_rule in ['adam', 'rmsprop']:
    print 'running with ', update_rule
    model = FullyConnectedNet([100, 100, 100, 100, 100], weight_scale=5e-2)
     
    solver = Solver(model, small_data,
                      num_epochs=5, batch_size=100,
                      update_rule=update_rule,
                      optim_config={
                        'learning_rate': learning_rates[update_rule]
                      },
                      verbose=True)
    solvers[update_rule] = solver
    solver.train()
    print
 
plt.subplot(3, 1, 1)
plt.title('Training loss')
plt.xlabel('Iteration')
 
plt.subplot(3, 1, 2)
plt.title('Training accuracy')
plt.xlabel('Epoch')
 
plt.subplot(3, 1, 3)
plt.title('Validation accuracy')
plt.xlabel('Epoch')
 
for update_rule, solver in solvers.iteritems():
    plt.subplot(3, 1, 1)
    plt.plot(solver.loss_history, 'o', label=update_rule)
   
    plt.subplot(3, 1, 2)
    plt.plot(solver.train_acc_history, '-o', label=update_rule)
 
    plt.subplot(3, 1, 3)
    plt.plot(solver.val_acc_history, '-o', label=update_rule)
   
for i in [1, 2, 3]:
    plt.subplot(3, 1, i)
    plt.legend(loc='upper center', ncol=4)
plt.gcf().set_size_inches(15, 15)
plt.show()

# running with  adam
# (Iteration 1 / 200) loss: 3.003398
# (Epoch 0 / 5) train acc: 0.102000; val_acc: 0.114000
# (Iteration 11 / 200) loss: 2.034254
# (Iteration 21 / 200) loss: 1.831142
# (Iteration 31 / 200) loss: 1.731255
# (Epoch 1 / 5) train acc: 0.360000; val_acc: 0.306000
# (Iteration 41 / 200) loss: 1.780875
# (Iteration 51 / 200) loss: 1.784328
# (Iteration 61 / 200) loss: 1.667602
# (Iteration 71 / 200) loss: 1.612633
# (Epoch 2 / 5) train acc: 0.414000; val_acc: 0.332000
# (Iteration 81 / 200) loss: 1.657031
# (Iteration 91 / 200) loss: 1.456176
# (Iteration 101 / 200) loss: 1.607429
# (Iteration 111 / 200) loss: 1.658154
# (Epoch 3 / 5) train acc: 0.504000; val_acc: 0.365000
# (Iteration 121 / 200) loss: 1.410839
# (Iteration 131 / 200) loss: 1.444539
# (Iteration 141 / 200) loss: 1.642703
# (Iteration 151 / 200) loss: 1.298852
# (Epoch 4 / 5) train acc: 0.567000; val_acc: 0.379000
# (Iteration 161 / 200) loss: 1.647283
# (Iteration 171 / 200) loss: 1.454301
# (Iteration 181 / 200) loss: 1.316260
# (Iteration 191 / 200) loss: 1.253265
# (Epoch 5 / 5) train acc: 0.581000; val_acc: 0.351000
# 
# running with  rmsprop
# (Iteration 1 / 200) loss: 2.653464
# (Epoch 0 / 5) train acc: 0.117000; val_acc: 0.122000
# (Iteration 11 / 200) loss: 2.021697
# (Iteration 21 / 200) loss: 1.893625
# (Iteration 31 / 200) loss: 1.823214
# (Epoch 1 / 5) train acc: 0.374000; val_acc: 0.316000
# (Iteration 41 / 200) loss: 1.774647
# (Iteration 51 / 200) loss: 1.988639
# (Iteration 61 / 200) loss: 1.788208
# (Iteration 71 / 200) loss: 1.687292
# (Epoch 2 / 5) train acc: 0.446000; val_acc: 0.314000
# (Iteration 81 / 200) loss: 1.723024
# (Iteration 91 / 200) loss: 1.784702
# (Iteration 101 / 200) loss: 1.492574
# (Iteration 111 / 200) loss: 1.505623
# (Epoch 3 / 5) train acc: 0.497000; val_acc: 0.344000
# (Iteration 121 / 200) loss: 1.436154
# (Iteration 131 / 200) loss: 1.590067
# (Iteration 141 / 200) loss: 1.524749
# (Iteration 151 / 200) loss: 1.413342
# (Epoch 4 / 5) train acc: 0.489000; val_acc: 0.352000
# (Iteration 161 / 200) loss: 1.610443
# (Iteration 171 / 200) loss: 1.471845
# (Iteration 181 / 200) loss: 1.497743
# (Iteration 191 / 200) loss: 1.384383
# (Epoch 5 / 5) train acc: 0.531000; val_acc: 0.348000

'''Train a good model'''
best_model = None
################################################################################
# TODO: Train the best FullyConnectedNet that you can on CIFAR-10. You might   #
# batch normalization and dropout useful. Store your best model in the         #
# best_model variable.                                                         #
################################################################################
X_val = data['X_val']
y_val= data['y_val']
X_test= data['X_test']
y_test= data['y_test']

learning_rate = 3.1e-4
weight_scale = 2.5e-2 #1e-5
model = FullyConnectedNet([600, 500, 400, 300, 200, 100],
                weight_scale=weight_scale, dtype=np.float64, dropout=0.25, use_batchnorm=True, reg=1e-2)
solver = Solver(model, data,
                print_every=500, num_epochs=30, batch_size=100,
                update_rule='adam',
                optim_config={
                  'learning_rate': learning_rate,
                },
                lr_decay=0.9
         )

solver.train()
scores = model.loss(data['X_test'])
y_pred = np.argmax(scores, axis = 1)
acc = np.mean(y_pred == data['y_test'])
print 'test acc: %f' %(acc)
best_model = model

plt.subplot(2, 1, 1)
plt.plot(solver.loss_history)
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(solver.train_acc_history, label='train')
plt.plot(solver.val_acc_history, label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.show() 


y_test_pred = np.argmax(best_model.loss(X_test), axis=1)
y_val_pred = np.argmax(best_model.loss(X_val), axis=1)
print 'Validation set accuracy: ', (y_val_pred == y_val).mean()
print 'Test set accuracy: ', (y_test_pred == y_test).mean()




