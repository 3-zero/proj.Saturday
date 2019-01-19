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

WIDTH = 300
HEIGHT = 300
LEV = 3
TRAIN = 100
TEST = 10

trXdata = []
trYdata = []
teXdata = []
teYdata = []
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
        #if( cnt >= TRAIN ) : break
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
        #if( cnt >= TEST ) : break
    rf.close()
    
train_data_set()
test_data_set()

print(len(trXdata))
print(len(teXdata))
#####******************************************************
# 4. Load pre-shuffled MNIST data into train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
# 5. Preprocess input data
#X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
#X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
#X_train /= 255
#X_test /= 255
# 6. Preprocess class labels
#Y_train = np_utils.to_categorical(y_train, 10)
#Y_test = np_utils.to_categorical(y_test, 10)
#####******************************************************

X_train = np.asarray(trXdata, dtype=np.float32)
X_test = np.asarray(teXdata, dtype=np.float32)

#print(X_train.shape[0])
#print(X_train.shape[1])
#print(X_test.shape[0])
#print(X_test.shape[1])

# 5. Preprocess input data
X_train = X_train.reshape(X_train.shape[0], WIDTH, HEIGHT, LEV)
X_test = X_test.reshape(X_test.shape[0], WIDTH, HEIGHT, LEV)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#y_train = np.asarray(trYdata, dtype=np.float32)
#y_test = np.asarray(teYdata, dtype=np.float32)

# 6. Preprocess class labels
#Y_train = np_utils.to_categorical(y_train, 2)
#Y_test = np_utils.to_categorical(y_test, 2)

Y_train = np.asarray(trYdata, dtype=np.float32)
Y_test = np.asarray(teYdata, dtype=np.float32)

#print("After Reshape")
#print(X_train.shape[0])
#print(X_train.shape[1])
#print(X_train.shape[2])
#print(X_train.shape[3])

#print(Y_train.shape[0])
#print(Y_train.shape[1])


del trXdata
del teXdata
del trYdata
del teYdata

#####******************************************************
# 7. Define model architecture
#model = Sequential()
 
#model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))
#model.add(Convolution2D(32, 3, 3, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.25))
 
#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(10, activation='softmax'))
#####******************************************************

model = Sequential()
 
    
#======================= 이 부분을 수정 =======================
#model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(WIDTH,HEIGHT,LEV)))
##print(model.output_shape)
#model.add(Convolution2D(32, (3, 3), activation='relu'))
##print(model.output_shape)
#model.add(MaxPooling2D(pool_size=(2,2)))
##print(model.output_shape)
#model.add(Dropout(0.25))
##print(model.output_shape)

#model.add(Flatten())
##print(model.output_shape)
#model.add(Dense(128, activation='relu')) # dense -hidden layer 기능!
##print(model.output_shape)
#model.add(Dropout(0.5))
##print(model.output_shape)
##model.add(Dense(10, activation='softmax'))
## 숫자 예측(10) --> 개/고양이 예측 (2)
#model.add(Dense(2, activation='softmax'))
##print(model.output_shape)
#=====================================================================

model.add(Convolution2D(50, (3,3), activation='relu', input_shape=(WIDTH,HEIGHT,LEV)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(100, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu')) # dense -hidden layer 기능!
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

          
#####******************************************************
# 8. Compile model
#model.compile(loss='categorical_crossentropy',
#              optimizer='adam',
#              metrics=['accuracy'])
 
# 9. Fit model on training data
#model.fit(X_train, Y_train, 
#          batch_size=32, nb_epoch=10, verbose=1)
 
# 10. Evaluate model on test data
#score = model.evaluate(X_test, Y_test, verbose=0)
#####******************************************************


# 8. Compile model
model.compile(loss='categorical_crossentropy',  optimizer='adam', metrics=['accuracy'])
 
# 9. Fit model on training data
model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)


# 10. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
