#encoding:utf-8
import random
import numpy as np
from cs231n.data_utils import get_CIFAR10_data
import matplotlib.pyplot as plt
from astropy.stats.histogram import histogram

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# CIFAR-10 Data Loading and Preprocessing
#Load the row CIFAR-10 data
X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev=get_CIFAR10_data()
print(X_train.shape,X_val.shape,X_test.shape,X_dev.shape)
print(y_train.shape,y_val.shape,y_test.shape,y_dev.shape)

# Extract Features
# For each image we will compute a Histogram of Oriented Gradients (HOG) as well as 
# a color histogram using the hue channel in HSV color space. 
# We form our final feature vector for each image by concatenating the HOG 
# and color histogram feature vectors.
# Roughly speaking, HOG should capture the texture of the image 
# while ignoring color information, and the color histogram represents 
# the color of the input image while ignoring texture. 
# As a result, we expect that using both together ought to work 
# better than using either alone. Verifying this assumption would be 
# a good thing to try for the bonus section.
# The hog_feature and color_histogram_hsv functions both operate on 
# a single image and return a feature vector for that image. 
# The extract_features function takes a set of images and a list of 
# feature functions and evaluates each feature function on each image, 
# storing the results in a matrix where each column is the concatenation of all feature vectors for a single image.

from cs231n.features import*
num_color_bins = 10 # Number of bins in the color histogram
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
X_train_feats = extract_features(X_train, feature_fns, verbose=True)
X_val_feats = extract_features(X_val, feature_fns)
X_test_feats = extract_features(X_test, feature_fns)

