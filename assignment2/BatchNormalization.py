# encoding: utf-8
'''
date:2017.07.31

Batch Normalization
One way to make deep networks easier to train is to use 
more sophisticated optimization procedures such as SGD+momentum, RMSProp, or Adam. 

Another strategy is to change the architecture of the network to make it easier to train. 
One idea along these lines is batch normalization which was recently proposed by [3].

The idea is relatively straightforward. Machine learning methods tend to work better 
when their input data consists of uncorrelated features with zero mean and unit variance. 

When training a neural network, we can preprocess the data 
before feeding it to the network to explicitly decorrelate its features; 
this will ensure that the first layer of the network sees data that follows a nice distribution. 

However even if we preprocess the input data, the activations at deeper layers of the network 
will likely no longer be decorrelated and will no longer have zero mean or unit variance 
since they are output from earlier layers in the network. 

Even worse, during the training process the distribution of features at each layer of the network 
will shift as the weights of each layer are updated.

The authors of [3] hypothesize that the shifting distribution of features 
inside deep neural networks may make training deep networks more difficult. 
To overcome this problem, [3] proposes to insert batch normalization layers into the network. 

At training time, a batch normalization layer uses a minibatch of data to 
estimate the mean and standard deviation of each feature. 

These estimated means and standard deviations are then used to center and normalize the features of the minibatch. 
A running average of these means and standard deviations is kept during training, 
and at test time these running averages are used to center and normalize features.

It is possible that this normalization strategy could reduce the representational power of the network, 
since it may sometimes be optimal for certain layers to have features that are not zero-mean or unit variance.
To this end, the batch normalization layer includes learnable shift and scale parameters for each feature dimension.

[3] Sergey Ioffe and Christian Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", ICML 2015.
'''

# As usual, a bit of setup

import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver


# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# Load the (preprocessed) CIFAR10 data.

data = get_CIFAR10_data()
for k, v in data.iteritems():
    print '%s: ' % k, v.shape

# X_val:  (1000L, 3L, 32L, 32L)
# X_train:  (49000L, 3L, 32L, 32L)
# X_test:  (1000L, 3L, 32L, 32L)
# y_val:  (1000L,)
# y_train:  (49000L,)
# y_test:  (1000L,)

'''Batch normalization: Forward'''
# Check the training-time forward pass by checking means and variances
# of features both before and after batch normalization

# Simulate the forward pass for a two-layer network
N, D1, D2, C = 200, 50, 60, 3
X = np.random.randn(N, D1)
W1 = np.random.randn(D1, D2)
W2 = np.random.randn(D2, C)
 
out = np.maximum(0, X.dot(W1)).dot(W2) # (N, C)
 
print 'Before batch normalization:'
print '  means: ', out.mean(axis=0)
print '  stds: ', out.std(axis=0)
 
# Means should be close to zero and stds close to one
# batchnorm_forward(x, gamma, beta, bn_param)
print 'After batch normalization (gamma=1, beta=0)'
out_norm, _ = batchnorm_forward(out, np.ones(C), np.zeros(C), {'mode': 'train'})
print '  mean: ', out_norm.mean(axis=0)
print '  std: ', out_norm.std(axis=0)
 
# Now means should be close to beta and stds close to gamma
gamma = np.asarray([1.0, 2.0, 3.0])
beta = np.asarray([11.0, 12.0, 13.0])
 
out_norm, _ = batchnorm_forward(out, gamma, beta, {'mode': 'train'})
print 'After batch normalization (nontrivial gamma, beta)'
print '  mean: ', out_norm.mean(axis=0)
print '  std: ', out_norm.std(axis=0)
 
# Before batch normalization:
#   means:  [-33.80810517 -16.12448246   9.57593977]
#   stds:  [ 29.556157    32.35463628  28.70940668]
# After batch normalization (gamma=1, beta=0)
#   mean:  [  1.48769885e-16  -1.40581990e-16   3.63598041e-17]
#   std:  [ 0.99999999  1.          0.99999999]
# After batch normalization (nontrivial gamma, beta)
#   mean:  [ 11.  12.  13.]
#   std:  [ 0.99999999  1.99999999  2.99999998]

# Check the test-time forward pass by running the training-time
# forward pass many times to warm up the running averages, and then
# checking the means and variances of activations after a test-time
# forward pass.

N, D1, D2, C = 200, 50, 60, 3
W1 = np.random.randn(D1, D2)
W2 = np.random.randn(D2, C)
 
bn_param = {'mode': 'train'}
gamma = np.ones(C)
beta = np.zeros(C)
 
for t in range(50):
    X = np.random.randn(N, D1)
    out = np.maximum(0, X.dot(W1)).dot(W2)
    batchnorm_forward(out, gamma, beta, bn_param)
 
bn_param['mode'] = 'test'
X = np.random.randn(N, D1)
out = np.maximum(0, X.dot(W1)).dot(W2)
out_norm, _ = batchnorm_forward(out, gamma, beta, bn_param)
 
# Means should be close to zero and stds close to one, but will be
# noisier than training-time forward passes.
print 'After batch normalization (test-time):'
print '  means: ', out_norm.mean(axis=0)
print '  stds: ', out_norm.std(axis=0) 

# After batch normalization (test-time):
#   means:  [-0.00460652  0.04272563 -0.09740718]
#   stds:  [ 0.95810544  0.95892265  1.04436518]

'''Batch Normalization: backward'''
# Gradient check batchnorm backward pass

N, D = 4, 5
x = 5 * np.random.randn(N, D) + 12
gamma = np.random.randn(D)
beta = np.random.randn(D)
dout = np.random.randn(N, D)
 
bn_param = {'mode': 'train'}
fx = lambda x: batchnorm_forward(x, gamma, beta, bn_param)[0]
fg = lambda a: batchnorm_forward(x, gamma, beta, bn_param)[0]
fb = lambda b: batchnorm_forward(x, gamma, beta, bn_param)[0]
 
dx_num = eval_numerical_gradient_array(fx, x, dout)
da_num = eval_numerical_gradient_array(fg, gamma, dout)
db_num = eval_numerical_gradient_array(fb, beta, dout)
 
_, cache = batchnorm_forward(x, gamma, beta, bn_param)
dx, dgamma, dbeta = batchnorm_backward(dout, cache)
print 'dx error: ', rel_error(dx_num, dx)
print 'dgamma error: ', rel_error(da_num, dgamma)
print 'dbeta error: ', rel_error(db_num, dbeta)

# dx error:  2.50576337205e-09
# dgamma error:  2.62793677432e-12
# dbeta error:  3.27551711715e-12

'''Batch Normalization: alternative backward'''
N, D = 100, 500
x = 5 * np.random.randn(N, D) + 12
gamma = np.random.randn(D)
beta = np.random.randn(D)
dout = np.random.randn(N, D)
 
bn_param = {'mode': 'train'}
out, cache = batchnorm_forward(x, gamma, beta, bn_param)
 
t1 = time.time()
dx1, dgamma1, dbeta1 = batchnorm_backward(dout, cache)
t2 = time.time()
dx2, dgamma2, dbeta2 = batchnorm_backward_alt(dout, cache)
t3 = time.time()
 
print 'dx difference: ', rel_error(dx1, dx2)
print 'dgamma difference: ', rel_error(dgamma1, dgamma2)
print 'dbeta difference: ', rel_error(dbeta1, dbeta2)
print 'speedup: %.2fx' % ((t2 - t1) / (t3 - t2))

# dx difference:  9.17602661598e-13
# dgamma difference:  0.0
# dbeta difference:  0.0
# speedup: 1.75x

'''Fully Connected Nets with Batch Normalization'''
N, D, H1, H2, C = 2, 15, 20, 30, 10
X = np.random.randn(N, D)
y = np.random.randint(C, size = N)
 
for reg in [0, 3.14]:
    print 'Running check with reg = ', reg
    model = FullyConnectedNet([H1, H2], input_dim = D, num_classes=C, 
        use_batchnorm=True, reg=reg, weight_scale=5e-2, dtype=np.float64)
     
    loss, grads = model.loss(X, y)
    print 'Initial loss: ', loss
     
    for name in sorted(grads):
        f = lambda _: model.loss(X, y)[0]
        grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
        print '%s relative error: %.2e' % (name, rel_error(grad_num, grads[name]))
    if reg == 0: print

# Running check with reg =  0
# Initial loss:  2.07316322311
# W1 relative error: 2.60e-04
# W2 relative error: 1.05e-06
# W3 relative error: 3.49e-10
# b1 relative error: 1.11e-08
# b2 relative error: 2.22e-03
# b3 relative error: 1.64e-10
# beta1 relative error: 1.39e-08
# beta2 relative error: 1.58e-09
# gamma1 relative error: 5.29e-09
# gamma2 relative error: 7.08e-10
# 
# Running check with reg =  3.14
# Initial loss:  6.8103191294
# W1 relative error: 1.22e-06
# W2 relative error: 1.70e-05
# W3 relative error: 6.49e-09
# b1 relative error: 1.39e-09
# b2 relative error: 1.11e-08
# b3 relative error: 3.99e-10
# beta1 relative error: 7.91e-09
# beta2 relative error: 1.20e-07
# gamma1 relative error: 1.18e-07
# gamma2 relative error: 5.49e-08


'''Batchnorm for deep networks'''
# Try training a very deep net with batchnorm

hidden_dims = [100, 100, 100, 100, 100]
 
num_train = 1000
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}
 
weight_scale = 2e-2
bn_model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=True)
model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=False)
 
bn_solver = Solver(bn_model, small_data,
                num_epochs=10, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=200)
bn_solver.train()
 
solver = Solver(model, small_data,
                num_epochs=10, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=200)
solver.train()

# (Iteration 1 / 200) loss: 2.332977
# (Epoch 0 / 10) train acc: 0.131000; val_acc: 0.138000
# (Epoch 1 / 10) train acc: 0.311000; val_acc: 0.292000
# (Epoch 2 / 10) train acc: 0.429000; val_acc: 0.333000
# (Epoch 3 / 10) train acc: 0.483000; val_acc: 0.305000
# (Epoch 4 / 10) train acc: 0.557000; val_acc: 0.333000
# (Epoch 5 / 10) train acc: 0.602000; val_acc: 0.321000
# (Epoch 6 / 10) train acc: 0.596000; val_acc: 0.282000
# (Epoch 7 / 10) train acc: 0.687000; val_acc: 0.311000
# (Epoch 8 / 10) train acc: 0.713000; val_acc: 0.315000
# (Epoch 9 / 10) train acc: 0.799000; val_acc: 0.332000
# (Epoch 10 / 10) train acc: 0.806000; val_acc: 0.332000
# (Iteration 1 / 200) loss: 2.302348
# (Epoch 0 / 10) train acc: 0.137000; val_acc: 0.124000
# (Epoch 1 / 10) train acc: 0.274000; val_acc: 0.249000
# (Epoch 2 / 10) train acc: 0.278000; val_acc: 0.247000
# (Epoch 3 / 10) train acc: 0.336000; val_acc: 0.252000
# (Epoch 4 / 10) train acc: 0.371000; val_acc: 0.303000
# (Epoch 5 / 10) train acc: 0.441000; val_acc: 0.313000
# (Epoch 6 / 10) train acc: 0.473000; val_acc: 0.319000
# (Epoch 7 / 10) train acc: 0.527000; val_acc: 0.341000
# (Epoch 8 / 10) train acc: 0.583000; val_acc: 0.321000
# (Epoch 9 / 10) train acc: 0.588000; val_acc: 0.331000
# (Epoch 10 / 10) train acc: 0.639000; val_acc: 0.351000

plt.subplot(3, 1, 1)
plt.title('Training loss')
plt.xlabel('Iteration')
 
plt.subplot(3, 1, 2)
plt.title('Training accuracy')
plt.xlabel('Epoch')
 
plt.subplot(3, 1, 3)
plt.title('Validation accuracy')
plt.xlabel('Epoch')
 
plt.subplot(3, 1, 1)
plt.plot(solver.loss_history, 'o', label='baseline')
plt.plot(bn_solver.loss_history, 'o', label='batchnorm')
 
plt.subplot(3, 1, 2)
plt.plot(solver.train_acc_history, '-o', label='baseline')
plt.plot(bn_solver.train_acc_history, '-o', label='batchnorm')
 
plt.subplot(3, 1, 3)
plt.plot(solver.val_acc_history, '-o', label='baseline')
plt.plot(bn_solver.val_acc_history, '-o', label='batchnorm')
   
for i in [1, 2, 3]:
  plt.subplot(3, 1, i)
  plt.legend(loc='upper center', ncol=4)
plt.gcf().set_size_inches(15, 15)
plt.show()


'''Batch normalization and initialization'''
# Try training a very deep net with batchnorm

hidden_dims = [50, 50, 50, 50, 50, 50, 50]

num_train = 1000
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

bn_solvers = {}
solvers = {}
weight_scales = np.logspace(-4, 0, num=20)
for i, weight_scale in enumerate(weight_scales):
    print 'Running weight scale %d / %d' % (i+1, len(weight_scales))
    bn_model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=True)
    model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=False)
    
    bn_solver = Solver(bn_model, small_data,
                  num_epochs=10, batch_size=50,
                  update_rule='adam',
                  optim_config={
                    'learning_rate': 1e-3,
                  },
                  verbose=False, print_every=200)
    bn_solver.train()
    bn_solvers[weight_scale] = bn_solver
    
    solver = Solver(model, small_data,
                num_epochs=10, batch_size=50,
                update_rule='adam',
                optim_config={
                    'learning_rate': 1e-3,
                },
                   verbose=False, print_every=200)
    solver.train()
    solvers[weight_scale] = solver

# Plot results of weight scale experiment
best_train_accs, bn_best_train_accs = [], []
best_val_accs, bn_best_val_accs = [], []
final_train_loss, bn_final_train_loss = [], []

for ws in weight_scales:
    best_train_accs.append(max(solvers[ws].train_acc_history))
    bn_best_train_accs.append(max(bn_solvers[ws].train_acc_history))
      
    best_val_accs.append(max(solvers[ws].val_acc_history))
    bn_best_val_accs.append(max(bn_solvers[ws].val_acc_history))
      
    final_train_loss.append(np.mean(solvers[ws].loss_history[-100:]))
    bn_final_train_loss.append(np.mean(bn_solvers[ws].loss_history[-100:]))

# 对坐标取对数：横坐标plt.semilogx()，纵坐标plt.semilogy()，横纵同时plt.loglog()
plt.subplot(3, 1, 1)
plt.title('Best val accuracy vs weight initialization scale')
plt.xlabel('Weight initialization scale')
plt.ylabel('Best val accuracy')
plt.semilogx(weight_scales, best_val_accs, '-o', label='baseline')
plt.semilogx(weight_scales, bn_best_val_accs, '-o', label='batchnorm')
plt.legend(ncol=2, loc='lower right')

plt.subplot(3, 1, 2)
plt.title('Best train accuracy vs weight initialization scale')
plt.xlabel('Weight initialization scale')
plt.ylabel('Best training accuracy')
plt.semilogx(weight_scales, best_train_accs, '-o', label='baseline')
plt.semilogx(weight_scales, bn_best_train_accs, '-o', label='batchnorm')
plt.legend()

plt.subplot(3, 1, 3)
plt.title('Final training loss vs weight initialization scale')
plt.xlabel('Weight initialization scale')
plt.ylabel('Final training loss')
plt.semilogx(weight_scales, final_train_loss, '-o', label='baseline')
plt.semilogx(weight_scales, bn_final_train_loss, '-o', label='batchnorm')
plt.legend()

plt.gcf().set_size_inches(10, 15)
plt.show()


