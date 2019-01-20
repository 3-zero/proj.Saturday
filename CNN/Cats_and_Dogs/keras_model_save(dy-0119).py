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

trXdata = []
trYdata = []
teXdata = []
teYdata = []

PrXdata = []

def train_data_set( ) :
    cnt = 0
    rf = open("C:/Users/DaYoung Lee/Desktop/1009/dataset/train.txt", 'r')
    for line in rf :
        if(len(line)<3):continue
        y = [0]*2
        y[int(line[-2])] = 1
        line = line[:-3].split(',')
        line = [float(item)/255.0 for item in line]
        trXdata.append(line)
        trYdata.append(y)
        cnt += 1
    rf.close()
    
def test_data_set() :
    cnt = 0
    rf = open("C:/Users/DaYoung Lee/Desktop/1009/dataset/test.txt", 'r')
    for line in rf :
        if(len(line)<3 ): continue
        y = [0]*2
        y[int(line[-2])] = 1
        line = line[:-3].split(',')
        line = [float(item)/255.0 for item in line]
        teXdata.append(line)
        teYdata.append(y)
        cnt += 1
    rf.close()
    
def predict_data_set() :
    file = "cat-test.jpg"
    im = Image.open("cat-test.jpg")
    
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
    
train_data_set()
test_data_set()

print("training data size:", len(trXdata))
print("test data size:", len(teXdata))

X_train = np.asarray(trXdata, dtype=np.float32)
X_test = np.asarray(teXdata, dtype=np.float32)

X_train = X_train.reshape(X_train.shape[0], WIDTH, HEIGHT, LEV)
X_test = X_test.reshape(X_test.shape[0], WIDTH, HEIGHT, LEV)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

Y_train = np.asarray(trYdata, dtype=np.float32)
Y_test = np.asarray(teYdata, dtype=np.float32)


del trXdata
del teXdata
del trYdata
del teYdata

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
 
model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)
print(score)

model.save_weights('cnn_cat_dog.h5')

predict_data_set()
P_train = np.asarray(PrXdata, dtype=np.float32)
P_train = P_train.reshape(1, WIDTH, HEIGHT, LEV)
P_train = P_train.astype('float32')

def predict() :
    model.load_weights('cnn_cat_dog.h5') 
    output = model.predict(P_train) 
    print("Answer :", np.argmax(output))

predict()