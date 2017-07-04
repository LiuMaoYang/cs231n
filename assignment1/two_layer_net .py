#encoding:utf-8
'''
Created on 2017年6月30日
@author: samsung

Implementing a Neural Network
In this exercise we will develop a neural network with fully-connected layers 
to perform classification, and test it out on the CIFAR-10 dataset.
'''
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.neural_net import TwoLayerNet
from psutil import net_connections

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Create a small net and some toy data to check your implementations.
# Note that we set the random seed for repeatable experiments.
input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

def init_toy_model():
    # if the seed was set, the same random number is generated each time
    np.random.seed(0)
    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return X, y

net = init_toy_model()
X, y=init_toy_data()

''' Forward pass: compute scores'''
scores = net.loss(X) #if y is None: return scores
print('Your scores:')
print(scores)
# print(' ')
# print('correct scores:')
# correct_scores = np.asarray([
#   [-0.81233741, -1.27654624, -0.70335995],
#   [-0.17129677, -1.18803311, -0.47310444],
#   [-0.51590475, -1.01354314, -0.8504215 ],
#   [-0.15419291, -0.48629638, -0.52901952],
#   [-0.00618733, -0.12435261, -0.15226949]])
# print(correct_scores)
# print(' ')
# # The difference should be very small. We get < 1e-7
# print('Difference between your scores and correct scores:')
# print(np.sum(np.abs(scores - correct_scores)))

# Your scores:
# [[-0.81233741 -1.27654624 -0.70335995]
#  [-0.17129677 -1.18803311 -0.47310444]
#  [-0.51590475 -1.01354314 -0.8504215 ]
#  [-0.15419291 -0.48629638 -0.52901952]
#  [-0.00618733 -0.12435261 -0.15226949]]
#  
# correct scores:
# [[-0.81233741 -1.27654624 -0.70335995]
#  [-0.17129677 -1.18803311 -0.47310444]
#  [-0.51590475 -1.01354314 -0.8504215 ]
#  [-0.15419291 -0.48629638 -0.52901952]
#  [-0.00618733 -0.12435261 -0.15226949]]
#  
# Difference between your scores and correct scores:
# 3.68027204961e-08

'''Forward pass: compute loss'''
loss, _ = net.loss(X, y, reg=0.1)
# correct_loss = 1.30378789133
# # should be very small, we get < 1e-12
# print('Difference between your loss and correct loss:')
# print(np.sum(np.abs(loss - correct_loss)))

# Difference between your loss and correct loss:
# 1.79856129989e-13

'''Backward pass'''
from cs231n.gradient_check import eval_numerical_gradient
 
# Use numeric gradient checking to check your implementation of the backward pass.
# If your implementation is correct, the difference between the numeric and
# analytic gradients should be less than 1e-8 for each of W1, W2, b1, and b2.
 
loss, grads = net.loss(X, y, reg=0.1)
 
# these should all be less than 1e-8 or so
for param_name in grads:
    f = lambda W: net.loss(X, y, reg=0.1)[0]
    param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
    print('%s max relative error: %e' % (param_name, np.sum(np.abs(param_grad_num, grads[param_name]))))

'''
Train the network
To train the network we will use stochastic gradient descent (SGD), 
similar to the SVM and Softmax classifiers. 
Look at the function TwoLayerNet.train and TwoLayerNet.predict 
You should achieve a training loss less than 0.2.
'''
net = init_toy_model()
stats = net.train(X, y, X, y,
            learning_rate=1e-1, reg=1e-5,
            num_iters=100, verbose=False)
print('Final training loss: ', stats['loss_history'][-1])
# ('Final training loss: ', 0.017149607938732093)
# 
# # plot the loss history
plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()

'''Load the CIFAR-10 data'''
from cs231n.data_utils import get_CIFAR10_data
X_train, y_train, X_val, y_val, X_test, y_test, _, _=get_CIFAR10_data(add_bias=False)
print(X_train.shape, X_val.shape, X_test.shape)
print(y_train.shape, y_val.shape, y_test.shape)

# ((49000L, 3072L), (1000L, 3072L), (1000L, 3072L))
# ((49000L,), (1000L,), (1000L,))

'''
Train a network
To train our network we will use SGD with momentum. 
In addition, we will adjust the learning rate with an exponential learning rate schedule 
as optimization proceeds; after each epoch,
we will reduce the learning rate by multiplying it by a decay rate.
'''
input_size = 32 * 32 *3
hidden_size = 50
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)
 
#Train the network
stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=1000, batch_size=200,
            learning_rate=1e-4, learning_rate_decay=0.95,
            reg=0.5, verbose=True)

# Predict on the validation set
val_acc = (net.predict(X_val) == y_val).mean()
print('Validation accuracy: ', val_acc)

# iteration 100 / 1000: loss 2.302627
# iteration 200 / 1000: loss 2.297971
# iteration 300 / 1000: loss 2.272143
# iteration 400 / 1000: loss 2.206691
# iteration 500 / 1000: loss 2.133257
# iteration 600 / 1000: loss 2.111150
# iteration 700 / 1000: loss 2.026495
# iteration 800 / 1000: loss 2.072627
# iteration 900 / 1000: loss 1.967342
# ('Validation accuracy: ', 0.28100000000000003)

'''
Debug the training
With the default parameters we provided above, you should get a validation accuracy of about 0.29 
on the validation set. This isn't very good.
One strategy for getting insight into what's wrong is to plot the loss function 
and the accuracies on the training and validation sets during optimization.
Another strategy is to visualize the weights that were learned in the first layer 
of the network. In most neural networks trained on visual data, 
the first layer weights typically show some visible structure when visualized.
'''
Plot the loss function and train / validation accuracies
plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')
 
plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.show()

from cs231n.vis_utils import visualize_grid
 
# Visualize the weights of the network
 
def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()
# 
# show_net_weights(net)

'''
Tune your hyperparameters
What's wrong?. Looking at the visualizations above, we see that the loss is decreasing 
more or less linearly, which seems to suggest that the learning rate may be too low.
Moreover, there is no gap between the training and validation accuracy, 
suggesting that the model we used has low capacity, and that we should increase its size. 
On the other hand, with a very large model we would expect to see more overfitting, 
which would manifest itself as a very large gap between the training and validation accuracy.
Tuning. Tuning the hyperparameters and developing intuition for how they affect 
the final performance is a large part of using Neural Networks, so we want you to get 
a lot of practice. Below, you should experiment with different values of the various 
hyperparameters, including hidden layer size, learning rate, numer of training epochs, and regularization strength. You might also consider tuning the learning rate decay, but you should be able to get good performance using the default value.
Approximate results. You should be aim to achieve a classification accuracy of 
greater than 48% on the validation set. Our best network gets over 52% on the validation set.
Experiment: You goal in this exercise is to get as good of a result on CIFAR-10 
as you can, with a fully-connected Neural Network. For every 1% above 52% on the Test set 
we will award you with one extra bonus point. Feel free implement your own techniques 
(e.g. PCA to reduce dimensionality, or adding dropout, or adding features to the solver, etc.).
'''

input_size = 32 * 32 *3
hidden_size = 50
num_classes = 10
#################################################################################
# TODO: Tune hyperparameters using the validation set. Store your best trained  #
# model in best_net.                                                            #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
# automatically like we did on the previous exercises.                          #
#################################################################################
hidden_size = [75, 100, 125]
results = {}
best_val_acc = 0
best_net = None
learning_rates = np.array([0.7, 0.8, 0.9, 1, 1.1])*1e-3
regularization_strengths = [0.75, 1, 1.25]

print('running')
for hs in hidden_size:
    for lr in learning_rates:
        for reg in regularization_strengths:
            print('.'),
            net = TwoLayerNet(input_size, hs, num_classes)
            # Train the network
            stats = net.train(X_train, y_train, X_val, y_val,
                            num_iters=1500, batch_size=200,
                            learning_rate=lr, learning_rate_decay=0.95,
                            reg= reg, verbose=False)
            val_acc = (net.predict(X_val) == y_val).mean()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_net = net
            results[(hs, lr, reg)] = val_acc
print(' ')
print('finished')
for hs, lr, reg in sorted(results):
    val_acc = results[(hs, lr, reg)]
    print('hs %d lr %e reg %e val accuracy: %f' % (hs, lr, reg,  val_acc))
    
print('best validation accuracy achieved during cross-validation: %f' % best_val_acc)
# running
# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  
# finished
# hs 75 lr 7.000000e-04 reg 7.500000e-01 val accuracy: 0.471000
# hs 75 lr 7.000000e-04 reg 1.000000e+00 val accuracy: 0.485000
# hs 75 lr 7.000000e-04 reg 1.250000e+00 val accuracy: 0.465000
# hs 75 lr 8.000000e-04 reg 7.500000e-01 val accuracy: 0.474000
# hs 75 lr 8.000000e-04 reg 1.000000e+00 val accuracy: 0.482000
# hs 75 lr 8.000000e-04 reg 1.250000e+00 val accuracy: 0.478000
# hs 75 lr 9.000000e-04 reg 7.500000e-01 val accuracy: 0.498000
# hs 75 lr 9.000000e-04 reg 1.000000e+00 val accuracy: 0.479000
# hs 75 lr 9.000000e-04 reg 1.250000e+00 val accuracy: 0.480000
# hs 75 lr 1.000000e-03 reg 7.500000e-01 val accuracy: 0.472000
# hs 75 lr 1.000000e-03 reg 1.000000e+00 val accuracy: 0.492000
# hs 75 lr 1.000000e-03 reg 1.250000e+00 val accuracy: 0.462000
# hs 75 lr 1.100000e-03 reg 7.500000e-01 val accuracy: 0.500000
# hs 75 lr 1.100000e-03 reg 1.000000e+00 val accuracy: 0.494000
# hs 75 lr 1.100000e-03 reg 1.250000e+00 val accuracy: 0.487000
# hs 100 lr 7.000000e-04 reg 7.500000e-01 val accuracy: 0.478000
# hs 100 lr 7.000000e-04 reg 1.000000e+00 val accuracy: 0.472000
# hs 100 lr 7.000000e-04 reg 1.250000e+00 val accuracy: 0.479000
# hs 100 lr 8.000000e-04 reg 7.500000e-01 val accuracy: 0.481000
# hs 100 lr 8.000000e-04 reg 1.000000e+00 val accuracy: 0.480000
# hs 100 lr 8.000000e-04 reg 1.250000e+00 val accuracy: 0.481000
# hs 100 lr 9.000000e-04 reg 7.500000e-01 val accuracy: 0.495000
# hs 100 lr 9.000000e-04 reg 1.000000e+00 val accuracy: 0.484000
# hs 100 lr 9.000000e-04 reg 1.250000e+00 val accuracy: 0.484000
# hs 100 lr 1.000000e-03 reg 7.500000e-01 val accuracy: 0.496000
# hs 100 lr 1.000000e-03 reg 1.000000e+00 val accuracy: 0.492000
# hs 100 lr 1.000000e-03 reg 1.250000e+00 val accuracy: 0.486000
# hs 100 lr 1.100000e-03 reg 7.500000e-01 val accuracy: 0.510000
# hs 100 lr 1.100000e-03 reg 1.000000e+00 val accuracy: 0.504000
# hs 100 lr 1.100000e-03 reg 1.250000e+00 val accuracy: 0.484000
# hs 125 lr 7.000000e-04 reg 7.500000e-01 val accuracy: 0.475000
# hs 125 lr 7.000000e-04 reg 1.000000e+00 val accuracy: 0.471000
# hs 125 lr 7.000000e-04 reg 1.250000e+00 val accuracy: 0.492000
# hs 125 lr 8.000000e-04 reg 7.500000e-01 val accuracy: 0.482000
# hs 125 lr 8.000000e-04 reg 1.000000e+00 val accuracy: 0.485000
# hs 125 lr 8.000000e-04 reg 1.250000e+00 val accuracy: 0.496000
# hs 125 lr 9.000000e-04 reg 7.500000e-01 val accuracy: 0.504000
# hs 125 lr 9.000000e-04 reg 1.000000e+00 val accuracy: 0.476000
# hs 125 lr 9.000000e-04 reg 1.250000e+00 val accuracy: 0.487000
# hs 125 lr 1.000000e-03 reg 7.500000e-01 val accuracy: 0.509000
# hs 125 lr 1.000000e-03 reg 1.000000e+00 val accuracy: 0.496000
# hs 125 lr 1.000000e-03 reg 1.250000e+00 val accuracy: 0.474000
# hs 125 lr 1.100000e-03 reg 7.500000e-01 val accuracy: 0.504000
# hs 125 lr 1.100000e-03 reg 1.000000e+00 val accuracy: 0.495000
# hs 125 lr 1.100000e-03 reg 1.250000e+00 val accuracy: 0.492000
# best validation accuracy achieved during cross-validation: 0.510000

# visualize the weights of the best network
show_net_weights(best_net)

'''Run on the test set'''
y_test_pre = best_net.predict(X_test)
test_acc = np.mean(y_test == y_test_pre)
print('Test accuracy:%d ' % test_acc)











