import numpy as np
np.random.seed(123)  # for reproducibility
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

import tensorflow as tf
import random
import os

import sys 
from keras import datasets, backend 
import matplotlib.pyplot as plt 
from PIL import Image 

WIDTH = 300
HEIGHT = 300
LEV = 3
TRAIN = 100
TEST = 10

PrXdata = []

def predict_data_set(filename) :
    global PrXdata
    PrXdata = []
    im = Image.open(filename)
    plt.imshow(im) 
    plt.show()
    if im.mode in ('RGBA', 'LA', 'P'): return
    im = im.resize((300,300), Image.ANTIALIAS)
    rgb_im = im.convert('RGB')
    for i in range(im.size[0]) :
        for j in range(im.size[1]) :
            L = rgb_im.getpixel((i, j))
            L = list(map(float, L))
            for k in L :
                PrXdata.append(k/255.0)

model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(WIDTH,HEIGHT,LEV)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu')) # dense -hidden layer 기능!
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',  optimizer='adam', metrics=['accuracy'])

def predict(filename) :
    predict_data_set(filename)
    P_train = np.asarray(PrXdata, dtype=np.float32)
    P_train = P_train.reshape(1, WIDTH, HEIGHT, LEV)
    P_train = P_train.astype('float32')
    model.load_weights('cnn_cat_dog.h5') 
    output = model.predict(P_train) 
    print("Output: ", output)
    print("Answer :", np.argmax(output))

predict("dog-test.jpg")
