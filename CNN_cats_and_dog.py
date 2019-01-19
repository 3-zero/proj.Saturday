import tensorflow as tf
from tensorflow import keras
import numpy as np


IMAGE_SIZE = 300
NUM_CHANNELS = 3
NUM_LABELS = 2
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 10
EVAL_BATCH_SIZE = 10

def load_data():
    WORK_DIRECTORY = './data'
    train_filename = WORK_DIRECTORY + '/test_2.txt'
    #test_filename = WORK_DIRECTORY + '/test_2.txt'

    train_X = []
    train_y = []

    train = open(train_filename, 'r')
    for line in train.readlines():
        line = line.strip()
        line = line.split(',')
        if len(line) < 3:
            continue
        for i in range(len(line)):
            line[i] = line[i].strip()
            line[i] = int(line[i])
        train_X.append(line[:len(line)-1])
        train_y.append([int(line[-1])])
    train.close()

    test_X = train_X
    test_y = train_y

    for i in range(len(train_X)):
        for j in range(len(train_X[i])):
            train_X[i][j] = train_X[i][j] / 255.0

    for i in range(len(test_X)):
        for j in range(len(test_X[i])):
            test_X[i][j] = test_X[i][j] / 255.0

    return train_X, train_y, test_X, test_y

if __name__ == '__main__':
    train_images, train_labels, test_X, test_y = load_data()
    train_X = np.array(train_images)
    train_X = tf.reshape(train_X, [len(train_images), IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])

    train_y = np.array(train_labels)
    train_y = tf.reshape(train_labels, [-1,1])

    input_shape=(300,300,3)

    print(train_X.shape)
    print(train_y.shape)


    #filter = kernel
    #kernel_size = (height, width, depth)
    #kernel_size's depth = # of channels


    model = keras.Sequential([
        keras.layers.Conv2D(filters=32, kernel_size=(5,5), strides=(1,1), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        keras.layers.Conv2D(filters=64, kernel_size=(5,5), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(units=1000, activation='relu'),
        keras.layers.Dense(units=1, activation='sigmoid)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_X, train_y, epochs=5, steps_per_epoch=1)
