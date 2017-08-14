# encoding: utf-8
'''
date:2017.08.02

Dropout
Dropout [1] is a technique for regularizing neural networks by randomly
setting some features to zero during the forward pass. 
In this exercise you will implement a dropout layer and modify your fully-connected network to optionally use dropout.

[1] Geoffrey E. Hinton et al, "Improving neural networks by preventing co-adaptation of feature detectors", arXiv 2012
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

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# %load_ext autoreload
# %autoreload 2

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

'''Dropout forward pass'''

x = np.random.randn(500, 500) + 10
 
for p in [0.3, 0.6, 0.75]:
    out, _ = dropout_forward(x, {'mode': 'train', 'p': p})
    out_test, _ = dropout_forward(x, {'mode': 'test', 'p': p})
 
    print 'Running tests with p = ', p
    print 'Mean of input: ', x.mean()
    print 'Mean of train-time output: ', out.mean()
    print 'Mean of test-time output: ', out_test.mean()
    print 'Fraction of train-time output set to zero: ', (out == 0).mean()
    print 'Fraction of test-time output set to zero: ', (out_test == 0).mean()
    print

# Running tests with p =  0.3
# Mean of input:  9.99995389144
# Mean of train-time output:  10.0093298007
# Mean of test-time output:  9.99995389144
# Fraction of train-time output set to zero:  0.299284
# Fraction of test-time output set to zero:  0.0
# 
# Running tests with p =  0.6
# Mean of input:  9.99995389144
# Mean of train-time output:  10.0292854014
# Mean of test-time output:  9.99995389144
# Fraction of train-time output set to zero:  0.59898
# Fraction of test-time output set to zero:  0.0
# 
# Running tests with p =  0.75
# Mean of input:  9.99995389144
# Mean of train-time output:  9.9825252974
# Mean of test-time output:  9.99995389144
# Fraction of train-time output set to zero:  0.750404
# Fraction of test-time output set to zero:  0.0

'''Dropout backward pass'''

x = np.random.randn(10, 10) + 10
dout = np.random.randn(*x.shape)
 
dropout_param = {'mode': 'train', 'p': 0.8, 'seed': 123}
out, cache = dropout_forward(x, dropout_param)
dx = dropout_backward(dout, cache)
dx_num = eval_numerical_gradient_array(lambda xx: dropout_forward(xx, dropout_param)[0], x, dout)
 
print 'dx relative error: ', rel_error(dx, dx_num)

# dx relative error:  1.89290756171e-11

'''Fully-connected nets with Dropout'''
N, D, H1, H2, C = 2, 15, 20, 30, 10
X = np.random.randn(N, D)
y = np.random.randint(C, size=(N, ))
 
for dp in [0, 0.25, 0.5]:
    print 'Running check with dropout = ', dp
    model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
                 dropout=dp, weight_scale=5e-2, dtype=np.float64, seed=123)
     
    loss, grads = model.loss(X, y)
    print 'Initial loss: ', loss
 
    for name in sorted(grads):
        f = lambda _: model.loss(X, y)[0]
        grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
        print '%s relative error: %.2e' % (name, rel_error(grad_num, grads[name]))
    print

# Running check with dropout =  0
# Initial loss:  2.30148690286
# W1 relative error: 4.39e-07
# W2 relative error: 2.36e-06
# W3 relative error: 4.95e-08
# b1 relative error: 4.96e-09
# b2 relative error: 1.69e-09
# b3 relative error: 1.20e-10
# 
# Running check with dropout =  0.25
# Initial loss:  2.30045745042
# W1 relative error: 3.17e-06
# W2 relative error: 3.66e-05
# W3 relative error: 1.13e-07
# b1 relative error: 9.45e-08
# b2 relative error: 2.97e-09
# b3 relative error: 1.43e-10
# 
# Running check with dropout =  0.5
# Initial loss:  2.30753476416
# W1 relative error: 3.03e-07
# W2 relative error: 7.45e-08
# W3 relative error: 4.01e-08
# b1 relative error: 3.03e-08
# b2 relative error: 2.07e-09
# b3 relative error: 1.21e-10

'''Regularization experiment'''
# Train two identical nets, one with dropout and one without

num_train = 500
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

solvers = {}
dropout_list = [0, 0.25, 0.5, 0.75, 0.8, 0.9, 0.99]

for dp in dropout_list:
    model = FullyConnectedNet([500], dropout=dp)
    print 'dropout: ', dp
    
    solver = Solver(model, small_data,
                  num_epochs=25, batch_size=100,
                  update_rule='adam',
                  optim_config={
                    'learning_rate': 5e-4,
                  },
                  verbose=False, print_every=100)
    solver.train()
    solvers[dp] = solver
 
    
# Plot train and validation accuracies of the two models

train_accs = []
val_accs = []

for dp in dropout_list:
    solver = solvers[dp]
    train_accs.append(solver.train_acc_history[-1])
    val_accs.append(solver.val_acc_history[-1])

plt.plot(3, 1, 1)
for dp in dropout_list:
    plt.plot(solvers[dp].train_acc_history, 'o', label='%.2f dropout' % dp)
plt.title('Train accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(ncol=2, loc='lower right')

plt.subplot(3, 1, 2)
for dp in dropout_choices:
    plt.plot(solvers[dp].val_acc_history, 'o', label='%.2f dropout' % dp)
plt.title('Val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(ncol=3, loc='lower right')

plt.gcf().set_size_inches(15, 15)
plt.show()

# dropout:  0
# (Iteration 1 / 125) loss: 8.009819
# (Epoch 0 / 25) train acc: 0.216000; val_acc: 0.161000
# (Epoch 1 / 25) train acc: 0.362000; val_acc: 0.227000
# (Epoch 2 / 25) train acc: 0.518000; val_acc: 0.250000
# (Epoch 3 / 25) train acc: 0.580000; val_acc: 0.246000
# (Epoch 4 / 25) train acc: 0.674000; val_acc: 0.296000
# (Epoch 5 / 25) train acc: 0.738000; val_acc: 0.286000
# (Epoch 6 / 25) train acc: 0.766000; val_acc: 0.283000
# (Epoch 7 / 25) train acc: 0.826000; val_acc: 0.282000
# (Epoch 8 / 25) train acc: 0.862000; val_acc: 0.272000
# (Epoch 9 / 25) train acc: 0.894000; val_acc: 0.296000
# (Epoch 10 / 25) train acc: 0.948000; val_acc: 0.296000
# (Epoch 11 / 25) train acc: 0.936000; val_acc: 0.300000
# (Epoch 12 / 25) train acc: 0.948000; val_acc: 0.287000
# (Epoch 13 / 25) train acc: 0.954000; val_acc: 0.288000
# (Epoch 14 / 25) train acc: 0.962000; val_acc: 0.305000
# (Epoch 15 / 25) train acc: 0.968000; val_acc: 0.295000
# (Epoch 16 / 25) train acc: 0.990000; val_acc: 0.316000
# (Epoch 17 / 25) train acc: 0.964000; val_acc: 0.288000
# (Epoch 18 / 25) train acc: 0.980000; val_acc: 0.288000
# (Epoch 19 / 25) train acc: 0.956000; val_acc: 0.289000
# (Epoch 20 / 25) train acc: 0.966000; val_acc: 0.310000
# (Iteration 101 / 125) loss: 0.234272
# (Epoch 21 / 25) train acc: 0.980000; val_acc: 0.286000
# (Epoch 22 / 25) train acc: 0.958000; val_acc: 0.292000
# (Epoch 23 / 25) train acc: 0.948000; val_acc: 0.293000
# (Epoch 24 / 25) train acc: 0.972000; val_acc: 0.294000
# (Epoch 25 / 25) train acc: 0.978000; val_acc: 0.275000
# dropout:  0.25
# (Iteration 1 / 125) loss: 10.943137
# (Epoch 0 / 25) train acc: 0.238000; val_acc: 0.171000
# (Epoch 1 / 25) train acc: 0.350000; val_acc: 0.223000
# (Epoch 2 / 25) train acc: 0.524000; val_acc: 0.238000
# (Epoch 3 / 25) train acc: 0.640000; val_acc: 0.249000
# (Epoch 4 / 25) train acc: 0.648000; val_acc: 0.283000
# (Epoch 5 / 25) train acc: 0.748000; val_acc: 0.280000
# (Epoch 6 / 25) train acc: 0.798000; val_acc: 0.293000
# (Epoch 7 / 25) train acc: 0.814000; val_acc: 0.275000
# (Epoch 8 / 25) train acc: 0.778000; val_acc: 0.245000
# (Epoch 9 / 25) train acc: 0.908000; val_acc: 0.291000
# (Epoch 10 / 25) train acc: 0.872000; val_acc: 0.301000
# (Epoch 11 / 25) train acc: 0.916000; val_acc: 0.297000
# (Epoch 12 / 25) train acc: 0.918000; val_acc: 0.288000
# (Epoch 13 / 25) train acc: 0.938000; val_acc: 0.276000
# (Epoch 14 / 25) train acc: 0.934000; val_acc: 0.284000
# (Epoch 15 / 25) train acc: 0.964000; val_acc: 0.302000
# (Epoch 16 / 25) train acc: 0.950000; val_acc: 0.288000
# (Epoch 17 / 25) train acc: 0.956000; val_acc: 0.315000
# (Epoch 18 / 25) train acc: 0.958000; val_acc: 0.291000
# (Epoch 19 / 25) train acc: 0.966000; val_acc: 0.284000
# (Epoch 20 / 25) train acc: 0.968000; val_acc: 0.268000
# (Iteration 101 / 125) loss: 0.632392
# (Epoch 21 / 25) train acc: 0.986000; val_acc: 0.284000
# (Epoch 22 / 25) train acc: 0.968000; val_acc: 0.295000
# (Epoch 23 / 25) train acc: 0.986000; val_acc: 0.303000
# (Epoch 24 / 25) train acc: 0.986000; val_acc: 0.296000
# (Epoch 25 / 25) train acc: 0.990000; val_acc: 0.313000
# dropout:  0.5
# (Iteration 1 / 125) loss: 9.953348
# (Epoch 0 / 25) train acc: 0.216000; val_acc: 0.181000
# (Epoch 1 / 25) train acc: 0.414000; val_acc: 0.270000
# (Epoch 2 / 25) train acc: 0.498000; val_acc: 0.251000
# (Epoch 3 / 25) train acc: 0.574000; val_acc: 0.297000
# (Epoch 4 / 25) train acc: 0.636000; val_acc: 0.289000
# (Epoch 5 / 25) train acc: 0.698000; val_acc: 0.275000
# (Epoch 6 / 25) train acc: 0.724000; val_acc: 0.251000
# (Epoch 7 / 25) train acc: 0.808000; val_acc: 0.268000
# (Epoch 8 / 25) train acc: 0.828000; val_acc: 0.308000
# (Epoch 9 / 25) train acc: 0.886000; val_acc: 0.298000
# (Epoch 10 / 25) train acc: 0.832000; val_acc: 0.291000
# (Epoch 11 / 25) train acc: 0.890000; val_acc: 0.279000
# (Epoch 12 / 25) train acc: 0.906000; val_acc: 0.280000
# (Epoch 13 / 25) train acc: 0.910000; val_acc: 0.320000
# (Epoch 14 / 25) train acc: 0.926000; val_acc: 0.326000
# (Epoch 15 / 25) train acc: 0.938000; val_acc: 0.296000
# (Epoch 16 / 25) train acc: 0.960000; val_acc: 0.281000
# (Epoch 17 / 25) train acc: 0.956000; val_acc: 0.302000
# (Epoch 18 / 25) train acc: 0.944000; val_acc: 0.320000
# (Epoch 19 / 25) train acc: 0.966000; val_acc: 0.309000
# (Epoch 20 / 25) train acc: 0.956000; val_acc: 0.311000
# (Iteration 101 / 125) loss: 0.967650
# (Epoch 21 / 25) train acc: 0.972000; val_acc: 0.304000
# (Epoch 22 / 25) train acc: 0.970000; val_acc: 0.321000
# (Epoch 23 / 25) train acc: 0.972000; val_acc: 0.333000
# (Epoch 24 / 25) train acc: 0.978000; val_acc: 0.319000
# (Epoch 25 / 25) train acc: 0.972000; val_acc: 0.318000
# dropout:  0.75
# (Iteration 1 / 125) loss: 15.695389
# (Epoch 0 / 25) train acc: 0.216000; val_acc: 0.154000
# (Epoch 1 / 25) train acc: 0.336000; val_acc: 0.221000
# (Epoch 2 / 25) train acc: 0.446000; val_acc: 0.276000
# (Epoch 3 / 25) train acc: 0.526000; val_acc: 0.280000
# (Epoch 4 / 25) train acc: 0.594000; val_acc: 0.315000
# (Epoch 5 / 25) train acc: 0.600000; val_acc: 0.317000
# (Epoch 6 / 25) train acc: 0.636000; val_acc: 0.299000
# (Epoch 7 / 25) train acc: 0.706000; val_acc: 0.293000
# (Epoch 8 / 25) train acc: 0.676000; val_acc: 0.291000
# (Epoch 9 / 25) train acc: 0.752000; val_acc: 0.318000
# (Epoch 10 / 25) train acc: 0.756000; val_acc: 0.313000
# (Epoch 11 / 25) train acc: 0.800000; val_acc: 0.318000
# (Epoch 12 / 25) train acc: 0.818000; val_acc: 0.292000
# (Epoch 13 / 25) train acc: 0.824000; val_acc: 0.310000
# (Epoch 14 / 25) train acc: 0.816000; val_acc: 0.313000
# (Epoch 15 / 25) train acc: 0.832000; val_acc: 0.310000
# (Epoch 16 / 25) train acc: 0.866000; val_acc: 0.330000
# (Epoch 17 / 25) train acc: 0.866000; val_acc: 0.331000
# (Epoch 18 / 25) train acc: 0.864000; val_acc: 0.322000
# (Epoch 19 / 25) train acc: 0.884000; val_acc: 0.325000
# (Epoch 20 / 25) train acc: 0.890000; val_acc: 0.339000
# (Iteration 101 / 125) loss: 6.509651
# (Epoch 21 / 25) train acc: 0.906000; val_acc: 0.315000
# (Epoch 22 / 25) train acc: 0.914000; val_acc: 0.330000
# (Epoch 23 / 25) train acc: 0.896000; val_acc: 0.296000
# (Epoch 24 / 25) train acc: 0.898000; val_acc: 0.295000
# (Epoch 25 / 25) train acc: 0.924000; val_acc: 0.321000
# dropout:  0.8
# (Iteration 1 / 125) loss: 16.710509
# (Epoch 0 / 25) train acc: 0.248000; val_acc: 0.194000
# (Epoch 1 / 25) train acc: 0.354000; val_acc: 0.236000
# (Epoch 2 / 25) train acc: 0.408000; val_acc: 0.268000
# (Epoch 3 / 25) train acc: 0.512000; val_acc: 0.263000
# (Epoch 4 / 25) train acc: 0.556000; val_acc: 0.275000
# (Epoch 5 / 25) train acc: 0.556000; val_acc: 0.308000
# (Epoch 6 / 25) train acc: 0.582000; val_acc: 0.301000
# (Epoch 7 / 25) train acc: 0.640000; val_acc: 0.286000
# (Epoch 8 / 25) train acc: 0.646000; val_acc: 0.287000
# (Epoch 9 / 25) train acc: 0.664000; val_acc: 0.289000
# (Epoch 10 / 25) train acc: 0.720000; val_acc: 0.292000
# (Epoch 11 / 25) train acc: 0.752000; val_acc: 0.324000
# (Epoch 12 / 25) train acc: 0.712000; val_acc: 0.316000
# (Epoch 13 / 25) train acc: 0.718000; val_acc: 0.303000
# (Epoch 14 / 25) train acc: 0.780000; val_acc: 0.307000
# (Epoch 15 / 25) train acc: 0.768000; val_acc: 0.315000
# (Epoch 16 / 25) train acc: 0.798000; val_acc: 0.330000
# (Epoch 17 / 25) train acc: 0.808000; val_acc: 0.312000
# (Epoch 18 / 25) train acc: 0.818000; val_acc: 0.317000
# (Epoch 19 / 25) train acc: 0.844000; val_acc: 0.341000
# (Epoch 20 / 25) train acc: 0.842000; val_acc: 0.338000
# (Iteration 101 / 125) loss: 8.995750
# (Epoch 21 / 25) train acc: 0.852000; val_acc: 0.316000
# (Epoch 22 / 25) train acc: 0.848000; val_acc: 0.323000
# (Epoch 23 / 25) train acc: 0.872000; val_acc: 0.317000
# (Epoch 24 / 25) train acc: 0.894000; val_acc: 0.316000
# (Epoch 25 / 25) train acc: 0.902000; val_acc: 0.317000
# dropout:  0.9
# (Iteration 1 / 125) loss: 25.872705
# (Epoch 0 / 25) train acc: 0.170000; val_acc: 0.128000
# (Epoch 1 / 25) train acc: 0.322000; val_acc: 0.262000
# (Epoch 2 / 25) train acc: 0.360000; val_acc: 0.256000
# (Epoch 3 / 25) train acc: 0.410000; val_acc: 0.264000
# (Epoch 4 / 25) train acc: 0.458000; val_acc: 0.280000
# (Epoch 5 / 25) train acc: 0.492000; val_acc: 0.308000
# (Epoch 6 / 25) train acc: 0.508000; val_acc: 0.306000
# (Epoch 7 / 25) train acc: 0.484000; val_acc: 0.290000
# (Epoch 8 / 25) train acc: 0.562000; val_acc: 0.281000
# (Epoch 9 / 25) train acc: 0.578000; val_acc: 0.283000
# (Epoch 10 / 25) train acc: 0.564000; val_acc: 0.279000
# (Epoch 11 / 25) train acc: 0.574000; val_acc: 0.297000
# (Epoch 12 / 25) train acc: 0.630000; val_acc: 0.314000
# (Epoch 13 / 25) train acc: 0.628000; val_acc: 0.321000
# (Epoch 14 / 25) train acc: 0.648000; val_acc: 0.315000
# (Epoch 15 / 25) train acc: 0.660000; val_acc: 0.321000
# (Epoch 16 / 25) train acc: 0.686000; val_acc: 0.316000
# (Epoch 17 / 25) train acc: 0.698000; val_acc: 0.313000
# (Epoch 18 / 25) train acc: 0.706000; val_acc: 0.322000
# (Epoch 19 / 25) train acc: 0.720000; val_acc: 0.315000
# (Epoch 20 / 25) train acc: 0.732000; val_acc: 0.329000
# (Iteration 101 / 125) loss: 17.031117
# (Epoch 21 / 25) train acc: 0.724000; val_acc: 0.325000
# (Epoch 22 / 25) train acc: 0.684000; val_acc: 0.322000
# (Epoch 23 / 25) train acc: 0.730000; val_acc: 0.326000
# (Epoch 24 / 25) train acc: 0.764000; val_acc: 0.317000
# (Epoch 25 / 25) train acc: 0.732000; val_acc: 0.314000
# dropout:  0.99
# (Iteration 1 / 125) loss: 64.061104
# (Epoch 0 / 25) train acc: 0.116000; val_acc: 0.105000
# (Epoch 1 / 25) train acc: 0.158000; val_acc: 0.155000
# (Epoch 2 / 25) train acc: 0.198000; val_acc: 0.179000
# (Epoch 3 / 25) train acc: 0.232000; val_acc: 0.216000
# (Epoch 4 / 25) train acc: 0.258000; val_acc: 0.234000
# (Epoch 5 / 25) train acc: 0.264000; val_acc: 0.231000
# (Epoch 6 / 25) train acc: 0.288000; val_acc: 0.221000
# (Epoch 7 / 25) train acc: 0.284000; val_acc: 0.220000
# (Epoch 8 / 25) train acc: 0.294000; val_acc: 0.217000
# (Epoch 9 / 25) train acc: 0.276000; val_acc: 0.202000
# (Epoch 10 / 25) train acc: 0.310000; val_acc: 0.214000
# (Epoch 11 / 25) train acc: 0.292000; val_acc: 0.212000
# (Epoch 12 / 25) train acc: 0.294000; val_acc: 0.245000
# (Epoch 13 / 25) train acc: 0.302000; val_acc: 0.254000
# (Epoch 14 / 25) train acc: 0.302000; val_acc: 0.265000
# (Epoch 15 / 25) train acc: 0.316000; val_acc: 0.271000
# (Epoch 16 / 25) train acc: 0.320000; val_acc: 0.270000
# (Epoch 17 / 25) train acc: 0.332000; val_acc: 0.251000
# (Epoch 18 / 25) train acc: 0.302000; val_acc: 0.255000
# (Epoch 19 / 25) train acc: 0.328000; val_acc: 0.265000
# (Epoch 20 / 25) train acc: 0.368000; val_acc: 0.276000
# (Iteration 101 / 125) loss: 67.419978
# (Epoch 21 / 25) train acc: 0.360000; val_acc: 0.288000
# (Epoch 22 / 25) train acc: 0.366000; val_acc: 0.284000
# (Epoch 23 / 25) train acc: 0.352000; val_acc: 0.284000
# (Epoch 24 / 25) train acc: 0.350000; val_acc: 0.285000
# (Epoch 25 / 25) train acc: 0.342000; val_acc: 0.297000
