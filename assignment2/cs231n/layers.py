#encoding: utf-8
# from builtins import range
import numpy as np
from scipy.stats._continuous_distns import dgamma_gen
from scipy.signal import convolve2d

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M) D = d1 * ...* dk
    - b: A numpy array of biases, of shape (M,)

    out = x * w + b

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
#     pass
    N = x.shape[0]
    x_rsp = x.reshape(N,-1)
    out = x_rsp.dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      
    out = x * w + b
    dout: dloss/dout
    dx: dloss/dx   dx = dloss/dout * dout/dx = dout * wT (N, M) * (M, D) -->(N, D) -->(N, d1,...dk)
    dw: dloss/dw   dw = xT * dout (D, N) * (N, M)--->(D, M) 
    db: dloss/db   db = dout.T * ones(N, )  (M, N) * (N, ) --->(M, )

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
#     pass
    N = x.shape[0]
    x_rsp = x.reshape(N, -1)
    dx = dout.dot(w.T).reshape(*x.shape)
    dw = (x_rsp.T).dot(dout)
    bb = np.sum(dout, axis = 0)
    db = (dout.T).dot(np.ones(N)) # np.ones(N) NOT np.ones((N, 1)) ?
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    
    out = max(0, x)
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
#     pass
    out = x * (x >= 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    
    out = max(0, x)
    The max gate routes the gradient. 
    Unlike the add gate which distributed the gradient unchanged to all its inputs, 
    the max gate distributes the gradient (unchanged) to exactly one of its inputs 
    (the input that had the highest value during the forward pass). 
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
#     pass
    dx = dout * (x >= 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.
    在网络的每一层输入的时候，又插入了一个归一化层，
    也就是先做一个归一化处理，然后再进入网络的下一层
    
    归一化原因在于神经网络学习过程本质就是为了学习数据分布，
    一旦训练数据与测试数据的分布不同，那么网络的泛化能力也大大降低；
    另外一方面，一旦每批训练数据的分布各不相同(batch 梯度下降)，
    那么网络就要在每次迭代都去学习适应不同的分布，这样将会大大降低网络的训练速度，
    
    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
        means should be close to beta and stds close to gamma
        
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
#         pass
        sample_mean = np.mean(x, axis = 0) # (D, )
        sample_var = np.var(x, axis = 0)
        x_norm = (x - sample_mean) / np.sqrt(sample_var + eps) # normalize
        out = gamma * x_norm + beta # scale and shift for each col
        cache = (gamma, x, sample_mean, sample_var, eps, x_norm)
        
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
#         pass
        scale = gamma / (np.sqrt(running_var + eps))
        out = x * scale + (beta - running_mean * scale)
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    
    mean = 1 / m * sum(x)
    var = 1 / m * sum((x - mean)^2)
    x_norm = (x - mean) / sqrt(var + eps)
    out = gamma * x_norm + beta
    
    dout = dloss / dout
    dx_norm = gamma * dout
    
    dx_norm / dvar = (-1/2) * (x - mean) * (var + eps)^(-3/2)
    dvar = dx_norm * dx_norm / dvar
    
    dvar / dmean = (-2 / m) * sum(x - mean)
    dx_norm / dmean = -1 / sqrt(var + eps) + dx_norm / dvar * dvar / dmean
    dmean / dx = 1
    dvar / dx = (2 / m) * sum(x - mean) * dmean / dx
    dx_norm / dx = (1 - dmean / dx) * (1 / sqrt(var + eps)) + dx_norm / dvar * dvar / dmean
    
    dx = dx_norm * 
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
#     pass
    gamma, x, u_b, sigma_squared_b, eps, x_hat = cache
    N = x.shape[0]
    
#     dgamma = np.sum(dout * x_norm, axis = 0)
#     dbeta = np.sum(dout, axis = 0)
    
    dx_1 = gamma * dout
    dx_2_b = np.sum((x - u_b) * dx_1, axis=0)
    dx_2_a = ((sigma_squared_b + eps) ** -0.5) * dx_1
    dx_3_b = (-0.5) * ((sigma_squared_b + eps) ** -1.5) * dx_2_b
    dx_4_b = dx_3_b * 1
    dx_5_b = np.ones_like(x) / N * dx_4_b
    dx_6_b = 2 * (x - u_b) * dx_5_b
    dx_7_a = dx_6_b * 1 + dx_2_a * 1
    dx_7_b = dx_6_b * 1 + dx_2_a * 1
    dx_8_b = -1 * np.sum(dx_7_b, axis=0)
    dx_9_b = np.ones_like(x) / N * dx_8_b
    dx_10 = dx_9_b + dx_7_a
    
    dgamma = np.sum(x_hat * dout, axis=0)
    dbeta = np.sum(dout, axis=0)
    dx = dx_10
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    
    mean = 1 / m * sum(x)
    var = 1 / m * sum((x - mean)^2)
    x_norm = (x - mean) / sqrt(var + eps)
    out = gamma * x_norm + beta
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
#     pass
    gamma, x, mean, var, eps, x_hat = cache
    N = x.shape[0]
    dx_hat = dout * gamma
    
    dgamma = np.sum(dout * x_hat, axis = 0)
    dbeta = np.sum(dout, axis = 0)
    
    dvar = np.sum(dx_hat * (x - mean) * -0.5 * np.power(var + eps, -1.5), axis = 0)
    dmean = np.sum(dx_hat * -1 / np.sqrt(var + eps), axis = 0) + dvar * np.sum(-2 * (x - mean), axis = 0) /N
    
    dx = dx_hat * 1 / np.sqrt(var + eps) + dvar * 2.0 * (x - mean) / N + dmean * 1.0 / N
    
#     dx_norm = dout * gamma
#     dgamma = np.sum(x_hat * dout, axis=0)
#     dbeta = np.sum(dout, axis=0)
#     
#     dvar = np.sum(dx_norm * (x - mean) * -0.5 * np.power((var + eps), -1.5), axis=0)
#     dvar_dmean = np.mean(-2 * (x - mean), axis=0)
#     dmean = np.sum(dx_norm * -1 / np.sqrt(var + eps), axis=0) +  dvar * dvar_dmean
#     dvar_dx = 2.0 / N * (x - mean)
#     dmean_dx = 1.0 / N
#     dx = dx_norm / np.sqrt(var + eps) + dvar * dvar_dx + dmean_dx
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.
    
    https://yq.aliyun.com/articles/68901
    Dropout是指按照一定的概率将其暂时从网络中丢弃
    以概率P舍弃部分神经元，其它神经元以概率q=1-p被保留，舍去的神经元的输出都被设置为零
    对于随机梯度下降来说，由于是随机丢弃，故而每一个mini-batch都在训练不同的网络
    dropout对防止过拟合提高效果很有效
    
    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
#         pass
        mask = (np.random.rand(*x.shape) >= p) / (1 - p)
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
#         pass
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
#         pass
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    
    H_out = 1 + (H + 2 * pad - HH) / stride
    W_out = 1 + (W + 2 * pad - WW) / stride
    
    out = np.zeros((N, F, H_out, W_out))
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
#     x = np.random.randn(3,3)
#     x_pad = np.pad(x, ((0,1), (0,1)), mode='constant', constant_values=0)
#     [[-1.59086504 -0.84091118  0.37341282]
#      [ 1.194581   -0.1747599   1.51281457]
#      [-1.28133937  0.7868255  -1.8747617 ]]
#     
#     [[-1.59086504 -0.84091118  0.37341282  0.        ]
#      [ 1.194581   -0.1747599   1.51281457  0.        ]
#      [-1.28133937  0.7868255  -1.8747617   0.        ]
#      [ 0.          0.          0.          0.        ]]
    ###########################################################################
#     pass

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), 'constant') # (N, C, H+2*P, W+2*P)
    for i in range(H_out):
        for j in range(W_out):
            mask = x_pad[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW] #(N, C, HH, WW)
            for k in range(F):
                out[:, k , i, j] = np.sum(mask * w[k, :, :, :], axis=(1,2,3))
    out = out + (b)[None, :, None, None]        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
#     pass
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    
    H_out = 1 + (H + 2 * pad - HH) / stride
    W_out = 1 + (W + 2 * pad - WW) / stride
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), 'constant') 
    
    dx = np.zeros_like(x)
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    db = np.sum(dout, axis = (0, 2, 3)) # (F, )
    
    for i in range(H_out):
        for j in range(W_out):
            mask = x_pad[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
            for k in range(F):
                dw[k ,: ,: ,:] += np.sum(mask * (dout[:, k, i, j])[:, None, None, None], axis=0)
            for n in range(N): #compute dx_pad
                dx_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += np.sum((w[:, :, :, :] * 
                                                 (dout[n, :, i, j])[:,None ,None, None]), axis=0)
    dx = dx_pad[:,:,pad:-pad,pad:-pad]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
#     pass
    N, C, H, W = x.shape
    HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    H_out = (H-HH)/stride+1
    W_out = (W-WW)/stride+1
    
    out = np.zeros((N, C, H_out, W_out))
    
    for i in range(H_out):
        for j in range(W_out):
            mask = x[:, :, i*stride : i*stride+HH, j*stride : j*stride+WW]
            out[:, :, i, j] = np.max(mask, axis = (2, 3))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
#     pass
    x, pool_param = cache
    N, C, H, W = x.shape
    HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    H_out = (H-HH)/stride+1
    W_out = (W-WW)/stride+1
    
    dx = np.zeros_like(x)
    
    for i in range(H_out):
        for j in range(W_out):
            mask = x[:, :, i*stride : i*stride+HH, j*stride : j*stride+WW]
            max_mask = np.max(mask, axis = (2, 3))
            binary_mask = (mask == (max_mask)[:,:,None,None])
            dx[:, :, i*stride : i*stride+HH, j*stride : j*stride+WW] += binary_mask * (dout[:,:,i,j])[:,:,None,None]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x (N, C)
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
#     margins = np.maximum(0, x - x[np.arange(N), y].reshape(-1,1) + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    
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
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
