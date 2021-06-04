import random
from concurrent import futures
from threading import Lock

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, concatenate
from keras.layers.core import Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm
from keras.regularizers import l2

# from keras.applications import InceptionV3
import augmentation
from main import create_cnn

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
imsize = 299

policy = augmentation.ImageNetPolicy()

lock = Lock()


def get_siamese_model_without_contrasitive(input_shape):
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    _, convolutional_net = create_cnn(299, 299, 3, filters=(32, 64, 64))
    convolutional_net.load_weights("myconv50b.h5")
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

    siamese_net.compile(loss="binary_crossentropy", metrics=[tf.keras.metrics.BinaryAccuracy()], optimizer=optimizer)

    # return the model
    return siamese_net


def getGoodNotGood():
    cnt = 0
    images_path = 'myhouse_dataset/myhouse_dataset'
    X_house_images1 = np.zeros((121, imsize, imsize, 3), dtype='uint32')
    X_house_images2 = np.zeros((121, imsize, imsize, 3), dtype='uint32')
    X_house_labels = np.zeros((121), dtype='uint32')
    d = 0.0
    for i in tqdm(range(121)):
        if random.random() >= 0.5:
            sample = cv2.imread(images_path + '/' + str(i) + '.PNG')
            imgs = cv2.resize(sample, (imsize, imsize))
            X_house_images1[cnt] = imgs
            sample = cv2.imread(images_path + '/' + str(i) + '_2.PNG')
            imgs = cv2.resize(sample, (imsize, imsize))
            X_house_images2[cnt] = imgs
            X_house_labels[cnt] = 1
            cnt += 1
        else:
            path = images_path + '/' + str(i) + '_2.PNG'
            sample = cv2.imread(path)
            imgs = cv2.resize(sample, (imsize, imsize))
            X_house_images1[cnt] = imgs
            sample = cv2.imread(images_path + '/' + str(i) + '.PNG')
            imgs = cv2.resize(sample, (imsize, imsize))
            X_house_images2[cnt] = imgs
            X_house_labels[cnt] = 0
            cnt += 1
            d += 1

    print("Images: ", cnt)
    print("Images: ", d / cnt)
    X_house_images1 = X_house_images1 / 255.0
    X_house_images2 = X_house_images2 / 255.0
    return X_house_images1, X_house_images2, X_house_labels


def getBuildingNotBuilding():
    cnt = 0
    images_path = 'myhouse_dataset/myhouse_dataset'
    X_house_images = np.zeros((121, imsize, imsize, 3), dtype='float32')
    X_house_labels = np.zeros((121), dtype='uint32')
    for i in tqdm(range(121)):
        sample = cv2.imread(images_path + '/' + str(i) + '.PNG')
        if random.random() >= 0.5:
            sample = cv2.imread(images_path + '/' + str(i) + '_2.PNG')
        imgs = cv2.resize(sample, (imsize, imsize))
        X_house_images[cnt] = imgs
        cnt += 1

    with open('buildings.txt', 'r', encoding='utf-8') as f:
        for i, v in enumerate(f.readline().split()):
            X_house_labels[i // 2] = int(v)
    X_house_images = X_house_images / 255.0
    return X_house_images, X_house_labels


def train_siam():
    model = get_siamese_model_without_contrasitive((imsize, imsize, 3))
    print(model.summary())
    X_house_images1, X_house_images2, X_house_labels = getGoodNotGood()

    split = train_test_split(X_house_images1, X_house_images2, X_house_labels, test_size=0.25, random_state=30)
    (Ximage_train1, Ximage_test1, Ximage_train2, Ximage_test2, y_train, y_test,) = split

    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('siam.hdf5', save_best_only=True, monitor='val_loss', mode='min')

    model.fit(x=[Ximage_train1, Ximage_train2], y=y_train, validation_data=([Ximage_test1, Ximage_test2], y_test),
              callbacks=[earlyStopping, mcp_save], epochs=50, batch_size=10)

    results = model.evaluate([Ximage_test1, Ximage_test2], y_test, batch_size=10)
    print('test loss, test acc:', results)
    results2 = model.evaluate([Ximage_train1, Ximage_train2], y_train, batch_size=10)
    print('train loss, train acc:', results2)
    model.save_weights('siamb.h5')
    return results[1]


def train_cnn():
    input = Input((imsize, imsize, 3))
    _, convolutional_net = create_cnn(299, 299, 3, filters=(32, 64, 64))
    convolutional_net.load_weights("myconv10b.h5")
    # tf.keras.utils.plot_model(convolutional_net, to_file="cnn.png", show_shapes=True)
    # convolutional_net = InceptionV3()
    encoded_image = convolutional_net(input)
    x = Dense(256, activation="relu", kernel_regularizer=l2(0.0001))(encoded_image)
    x = Dense(64, activation="relu", kernel_regularizer=l2(0.0001))(x)

    # Same class or not prediction
    prediction = Dense(units=1, activation='sigmoid', kernel_regularizer=l2(0.0001))(x)

    # Connect the inputs with the outputs
    cnn_net = Model(inputs=input, outputs=prediction)
    optimizer = optimizers.Adam(lr=10e-4)

    cnn_net.compile(loss="binary_crossentropy", metrics=[tf.keras.metrics.BinaryAccuracy()], optimizer=optimizer)

    X_house_images, X_house_labels = getBuildingNotBuilding()
    # X_house_images, X_house_labels = augmented_dataset(X_house_images, X_house_labels)

    split = train_test_split(X_house_images, X_house_labels, test_size=0.25, random_state=42)
    (Ximage_train, Ximage_test, y_train, y_test,) = split
    Ximage_train, y_train = augmented_dataset(Ximage_train, y_train, False)
    batch_size = 10
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('best_aug.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    # reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')
    cnn_net.fit(x=Ximage_train, y=y_train, validation_data=(Ximage_test, y_test),
                callbacks=[earlyStopping, mcp_save], epochs=30, batch_size=batch_size)
    print(cnn_net.metrics_names)
    # cnn_net.load_weights('best_aug.hdf5')
    results = cnn_net.evaluate(Ximage_test, y_test, batch_size=batch_size)
    print('test loss, test acc:', results)
    results2 = cnn_net.evaluate(Ximage_train, y_train, batch_size=batch_size)
    print('train loss, train acc:', results2)
    return results[1]


def apply_affine_distortions(img, label):
    num_examples = 5
    array = np.zeros((num_examples, imsize, imsize, 3), dtype='float32')
    for j in range(num_examples):
        img2 = Image.fromarray((img * 255).astype('uint8'), 'RGB')
        transformed = policy(img2)
        transformed = policy(transformed)
        img2 = np.array(transformed)
        array[j] = img2 / 255
    return array, np.full(num_examples, label)


def augmented_dataset(X_house_images, X_house_labels, save=False):
    n = len(X_house_images)
    # print(np.array2string(X_house_labels))
    savepath = "augmented_dataset/"
    with futures.ProcessPoolExecutor() as executor:
        todo = []
        for i in range(n):
            future = executor.submit(apply_affine_distortions, X_house_images[i], X_house_labels[i])
            todo.append(future)
        for future in tqdm(futures.as_completed(todo)):
            lock.acquire()
            X_house_images = np.append(X_house_images, future.result()[0], axis=0)
            X_house_labels = np.append(X_house_labels, future.result()[1], axis=0)
            lock.release()
    if save:
        for i, img in enumerate(X_house_images):
            path = savepath + 'images/img' + str(i) + ".png"
            cv2.imwrite(path, img * 255)
        with open(savepath + "classes.txt", "w") as f:
            f.write(np.array2string(X_house_labels))
    X_house_images, X_house_labels = shuffle(X_house_images, X_house_labels, random_state=10)
    return X_house_images, X_house_labels


if __name__ == "__main__":
    # train_siam()
    train_cnn()
    su = 0
    for i in range(0, 20):
        su += train_cnn()
    print(su / 20)
