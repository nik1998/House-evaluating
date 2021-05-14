# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from keras import Model
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, concatenate
from tensorflow.keras import Input
import os
import cv2
from tqdm import tqdm


def create_ann(dim, regress=False):
    # define our MLP network
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation="relu"))
    model.add(Dense(4, activation="relu"))
    # check to see if the regression node should be added
    # return our model
    return model


convnet = None


def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
    global convnet
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
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
    x = Conv2D(filters[1], (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters[2], (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)

    x = Flatten()(x)
    convnet = Model(inputs, x)

    x2 = Dense(16)(convnet.output)
    x2 = Activation("relu")(x2)
    x2 = BatchNormalization(axis=chanDim)(x2)
    x2 = Dropout(0.5)(x2)

    x2 = Dense(4)(x2)
    x2 = Activation("relu")(x2)
    # construct the CNN
    model = Model(convnet.input, x2)
    # return the CNN
    model.summary()
    convnet.summary()
    return model


class My_Custom_Generator(keras.utils.Sequence):

    def __init__(self, image_filenames, image_attributes, labels, batch_size):
        self.image_filenames = image_filenames
        self.image_attributes = image_attributes
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        attrs = self.image_attributes[idx * self.batch_size: (idx + 1) * self.batch_size]
        return [attrs,np.array([
            cv2.resize(cv2.imread(file_name), (299, 299))
            for file_name in batch_x]) / 255.0], np.array(batch_y)


imsize = 299
images_path = 'archive/socal2/socal_pics'
X_house_images = []
for i in tqdm(range(15474)):
    X_house_images.append(images_path + '/' + str(i) + '.jpg')

df = pd.read_csv('archive/socal2.csv')
X_house_attributes = df[['n_citi', 'bed', 'bath', 'sqft', 'price']]
print(X_house_attributes)
print(X_house_attributes.shape)

bm = max(X_house_attributes['bed'])
sqftm = max(X_house_attributes['sqft'])
pricem = max(X_house_attributes['price'])
bathm = max(X_house_attributes['bath'])
citim = max(X_house_attributes['n_citi'])
X_house_attributes['n_citi'] = X_house_attributes['n_citi'] / citim
X_house_attributes['bed'] = X_house_attributes['bed'] / bm
X_house_attributes['sqft'] = X_house_attributes['sqft'] / sqftm
X_house_attributes['bath'] = X_house_attributes['bath'] / bathm
X_house_attributes['price'] = X_house_attributes['price'] / pricem
X1_final = np.zeros(4, dtype='float32')

from sklearn.model_selection import train_test_split

split = train_test_split(X_house_attributes, X_house_images, test_size=0.25, random_state=42)
(Xatt_train, Xatt_test, Ximage_train, Ximage_test) = split

y_train, y_test = Xatt_train['price'].values, Xatt_test['price'].values

X1_train = Xatt_train[['n_citi', 'bed', 'bath', 'sqft']].values
X2_train = Ximage_train
X1_test = Xatt_test[['n_citi', 'bed', 'bath', 'sqft']].values
X2_test = Ximage_test

print(X1_train.shape)
print(X1_test.shape)
print(y_train.shape)
print(y_test.shape)

# create the MLP and CNN models
mlp = create_ann(X1_train.shape[1], regress=False)
cnn = create_cnn(imsize, imsize, 3, filters=(32, 64, 64), regress=False)
# create the input to our final set of layers as the *output* of both
# the MLP and CNN
combinedInput = concatenate([mlp.output, cnn.output])
# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(4, activation="relu")(combinedInput)
x = Dense(1, activation="linear")(x)

from keras.optimizers import Adam

model = Model(inputs=[mlp.input, cnn.input], outputs=x)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
# train the model
model.summary()
convnet.summary()

model.compile(loss="mse", optimizer=opt)
for i in range(100):
    generator = My_Custom_Generator(X2_train,X1_train, y_train,10)
    # fit for one epoch
    model.fit_generator(generator, epochs=1, steps_per_epoch=1500, verbose=True)
    if i!= 0 and i% 10==0:
        convnet.save_weights('myconv'+str(i)+'b.h5')
# model.fit(x=[X1_train, X2_train], y=y_train, validation_data=([X1_test, X2_test], y_test), epochs=100, batch_size=64)

model.save_weights('myModel10b.h5')
convnet.save_weights('myconv10b.h5')
image_sample = cv2.imread('archive/socal2/socal_pics/4.jpg')
sample_resized = cv2.resize(image_sample, (imsize, imsize))
plt.imshow(sample_resized)

attr_sample = df.loc[df['image_id'] == 4]
print(attr_sample)

X1_final[0] = attr_sample['n_citi'] / citim
X1_final[1] = attr_sample['bed'] / bm
X1_final[2] = attr_sample['bath'] / bathm
X1_final[3] = attr_sample['sqft'] / sqftm
y_ground_truth = attr_sample['price']
X2_final = sample_resized / 255.0
print(X1_final.shape, " ", X2_final.shape)
y_pred = model.predict([np.reshape(X1_final, (1, 4)), np.reshape(X2_final, (1, imsize, imsize, 3))])
print("Actual price: ", attr_sample['price'].values)
print("Predicted price: ", y_pred * pricem)
# For example, here's several helpful packages to load
