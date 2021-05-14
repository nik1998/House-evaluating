import sys
import numpy as np
import pandas as pd
from matplotlib.pyplot import imread
import pickle
from os import listdir
import matplotlib.pyplot as plt

import cv2
import time

import tensorflow as tf
import random
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from tqdm import tqdm
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform

from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K

from sklearn.utils import shuffle

import numpy.random as rng
from keras import optimizers
from keras.applications.inception_v3 import InceptionV3


def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
    global convnet
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    # inpute shape: (64,64,3)
    inputShape = (height, width, depth)
    chanDim = -1
    # define the model input
    inputs = Input(shape=inputShape)
    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Conv2D(filters[0], (3, 3), padding="same")(inputs)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters[1], (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters[2], (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    convnet = Model(inputs, x)
    convnet.load_weights("myconv90b.h5")
    return convnet


def get_siamese_model(input_shape):
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    convolutional_net = create_cnn(299, 299, 3, filters=(32, 64, 64), regress=False)
    # convolutional_net = InceptionV3()
    # Generate the encodings (feature vectors) for the two images
    encoded_image_1 = convolutional_net(left_input)
    encoded_image_2 = convolutional_net(right_input)

    # L1 distance layer between the two encoded outputs
    # One could use Subtract from Keras, but we want the absolute value
    combinedInput = concatenate([encoded_image_1, encoded_image_2])

    x = Dense(512, activation="relu")(combinedInput)
    x = Dense(128, activation="relu")(x)

    # Same class or not prediction
    prediction = Dense(units=1, activation='sigmoid')(x)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
    optimizer = optimizers.Adam(lr=10e-4)

    siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer)

    # return the model
    return siamese_net

imsize = 299

model = get_siamese_model((imsize, imsize, 3))
print(model.summary())

cnt = 0
images_path = 'D:/mlprojects/differenceTwoImages/googlePhotos/mydataset'
X_house_images1 = np.zeros((121, imsize, imsize, 3), dtype='uint32')
X_house_images2 = np.zeros((121, imsize, imsize, 3), dtype='uint32')
X_house_labels = np.zeros((121), dtype='uint32')
d = 0.0
for i in tqdm(range(121)):
    if random.random() >= 0.5:
        sample = cv2.imread(images_path + '/' + str(i) + '.png')
        imgs = cv2.resize(sample, (imsize, imsize))
        X_house_images1[cnt] = imgs
        sample = cv2.imread(images_path + '/' + str(i) + '_2.png')
        imgs = cv2.resize(sample, (imsize, imsize))
        X_house_images2[cnt] = imgs
        X_house_labels[cnt] = 1
        cnt += 1
    else:
        path = images_path + '/' + str(i) + '_2.png'
        sample = cv2.imread(path)
        imgs = cv2.resize(sample, (imsize, imsize))
        X_house_images1[cnt] = imgs
        sample = cv2.imread(images_path + '/' + str(i) + '.png')
        imgs = cv2.resize(sample, (imsize, imsize))
        X_house_images2[cnt] = imgs
        X_house_labels[cnt] = 0
        cnt += 1
        d += 1

print("Images: ", cnt)
print("Images: ", d / cnt)

X_house_images1 = X_house_images1 / 255.0
X_house_images2 = X_house_images2 / 255.0
from sklearn.model_selection import train_test_split

split = train_test_split(X_house_images1, X_house_images2, X_house_labels, test_size=0.25, random_state=42)
(Ximage_train1, Ximage_test1, Ximage_train2, Ximage_test2, y_train, y_test,) = split

print(Ximage_train1.shape)
print(Ximage_test1.shape)
print(Ximage_train2.shape)
print(Ximage_test2.shape)
print(y_train.shape)
print(y_test.shape)

model.fit(x=[Ximage_train1, Ximage_train2], y=y_train, validation_data=([Ximage_test1, Ximage_test2], y_test),
          epochs=100, batch_size=10)

results = model.evaluate([Ximage_test1, Ximage_test2], y_test, batch_size=10)
print('test loss, test acc:', results)
results = model.evaluate([Ximage_train1, Ximage_train2], y_train, batch_size=10)
print('test loss, test acc:', results)
print(model.predict([Ximage_train1, Ximage_train2]))
