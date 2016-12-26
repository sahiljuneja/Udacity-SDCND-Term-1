### Import
import pickle
import cv2
import math
import time
import h5py
import json
import os
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils


### Load data
#TODO Improve the following if possible.

# Read in CSV file
csv_loc = "data/driving_log.csv"
df = pd.read_csv(csv_loc)

# Add c,l and r images.
features_col = pd.concat([df['center'], df['left'].map(str.strip), df['right'].map(str.strip)])
features_col = np.array(features_col.values.tolist())

# Add steering angles for c,l,r with added shift for l and r images
l_shift = 0.25
r_shift = -0.25
labels_c = df['steering']
labels_r = df['steering'] + r_shift
labels_l = df['steering'] + l_shift
labels_col = pd.concat([labels_c, labels_l, labels_r])
labels_col = np.array(labels_col.values.tolist())

print("Length of Features: {0}, Labels: {1}".format(len(features_col), len(labels_col)))

# Split csv data
features_col, labels_col = shuffle(features_col, labels_col)
X_train, X_val, y_train, y_val = train_test_split(features_col, labels_col, test_size=0.15, random_state=42232) 

# Read in image list
images = os.listdir("data/IMG/")

### Pre-Process
img_cols = 64
img_rows = 64

def data_trans(image, label):
    trans_factor = 100*random.random() - 50 # Parameters set based on original image dimensions
    trans_matrix = np.float32([[1,0,trans_factor],[0,1,trans_factor]])
    image_trans = cv2.warpAffine(image,trans_matrix,(image.shape[0],image.shape[1]))
    
    label = label + trans_factor/25
    
    return image_tr,steer_ang
    
    
def preprocess_data(image, label):
    # Translate image and steering angle
    image, label = data_trans(image, label)
    # Crop and resize
    image = image[60:140,40:280]
    image = cv2.resize(image, (img_cols, img_rows))
    
    # Normalize
    image = cv2.normalize(image, None, alpha=-0.5, beta=0.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    return image, label

### Helper Functions

def image_generator(csv_features, csv_labels):
    csv_features, csv_labels = shuffle(csv_features, csv_labels)
    
    for idx in range(len(csv_features)):
        p = random.random()
        label = csv_labels[idx]
        
        image = mpimg.imread("data/" + csv_features[idx])
        image, label = preprocess_data(image, label)
        
        yield image, label

epoch_labels = []
def train_data_generator(csv_features, csv_labels, batch_size):
    num_rows = int(len(csv_features))
    ctr = None
    batch_x = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_y = np.zeros(batch_size)
    while True:
        
        for i in range(batch_size):
            
            if ctr is None or ctr >= num_rows:
                print("length of batch: {0}".format(len(batch_x)))
                ctr = 0
                images = image_generator(csv_features, csv_labels)
            batch_x[i], batch_y[i] = next(images)
            ctr += 1

        yield (batch_x, batch_y)

def valid_data_generator(csv_features, csv_labels, batch_size):
    num_rows = int(len(csv_features))
    ctr = None
    batch_x = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_y = np.zeros(batch_size)
    while True:
        
        for i in range(batch_size):
            
            if ctr is None or ctr >= num_rows:
                print("length of batch: {0}".format(len(batch_x)))
                ctr = 0
                images = image_generator(csv_features, csv_labels)
            batch_x[i], batch_y[i] = next(images)
            ctr += 1
        
        yield (batch_x, batch_y)
        
print(X_train.shape)
	
### Parameters
layer_1_depth = 32
layer_2_depth = 48
layer_3_depth = 64
filter_size_1 = 3
filter_size_2 = 3
num_neurons_1 = 512
num_neurons_2 = 128
epochs = 3
batch_size = 64
samples_per_epoch = X_train.shape[0]
 
### Model
model = Sequential()
model.add(Convolution2D(layer_1_depth, filter_size_1, filter_size_1, border_mode = 'valid', subsample = (1,1), input_shape = (img_rows, img_cols, 3)))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Convolution2D(layer_2_depth, filter_size_1, filter_size_1, border_mode = 'valid', subsample = (1,1)))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Convolution2D(layer_3_depth, filter_size_1, filter_size_1, border_mode = 'valid', subsample = (1,1)))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(num_neurons_1))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(num_neurons_2))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(32))
model.add(Activation('elu'))
model.add(Dense(1))

model.summary()

model.compile(loss='mse',
              optimizer=Adam(lr = 0.0001),
              metrics=['mean_absolute_error'])

### Save Model
with open('model.json', 'w') as f:
	json.dump(model.to_json(), f)                                           

history = model.fit_generator(train_data_generator(X_train, y_train, batch_size), 
											samples_per_epoch=samples_per_epoch, 
											nb_epoch = epochs,
											verbose = 1,
											validation_data = valid_data_generator(X_val, y_val, batch_size),
                                                                                        nb_val_samples=X_val.shape[0])


### Save weights
model.save_weights('model.h5')
