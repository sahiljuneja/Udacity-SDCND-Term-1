### Import
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import cv2
import math
import time

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils


### Load data

### Split data
X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.15, random_state=432422)