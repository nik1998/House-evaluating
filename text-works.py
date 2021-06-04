import json
import os
import string
from keras import backend as K
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Lambda
from keras.layers.merge import add
from keras.models import Model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model
from keras.utils import to_categorical
from numpy import array
from numpy import argmax
from tensorflow.python.keras.callbacks import ModelCheckpoint
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import numpy as np
from main import create_cnn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from augmentation import text_init, random_text_aug
from siam_main import apply_affine_distortions
from nltk.stem import WordNetLemmatizer
import time
from collections import defaultdict
from keras.regularizers import l2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_siamese_model(input_shape):
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    _, convolutional_net = create_cnn(299, 299, 3, filters=(32, 64, 64))
    # Generate the encodings (feature vectors) for the two images
    encoded_image_1 = convolutional_net(left_input)
    encoded_image_2 = convolutional_net(right_input)

    # L1 distance layer between the two encoded outputs
    # One could use Subtract from Keras, but we want the absolute value
    l1_distance_layer = Lambda(
        lambda tensors: K.abs(tensors[0] - tensors[1]))
    l1_distance = l1_distance_layer([encoded_image_1, encoded_image_2])

    # Same class or not prediction
    prediction = Dense(units=128, activation='relu')(l1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
    optimizer = optimizers.Adam(lr=10e-4)

    siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer)

    # return the model
    return siamese_net


# load doc into memory
def load_doc():
    # open the file as read only
    mapping = dict()
    arr = json.loads(open('myhouse_dataset/annotations.json').read())
    for i in range(len(arr)):
        image_id = arr[i]['img_id']
        image_desc = arr[i]['sentences']
        if image_id not in mapping:
            mapping[image_id] = list()
        if image_desc not in mapping[image_id]:
            mapping[image_id].append(image_desc)

    return mapping


# extract descriptions for images
def load_descriptions(doc):
    mapping = dict()
    # process lines
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        if len(line) < 2:
            continue
        # take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        # remove filename from image id
        image_id = image_id.split('.')[0]
        # convert description tokens back to string
        image_desc = ' '.join(image_desc)
        # create the list if needed
        if image_id not in mapping:
            mapping[image_id] = list()
        # store description
        mapping[image_id].append(image_desc)
    return mapping


def clean_descriptions(descriptions):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    # nlp = spacy.load("en_core_web_sm")
    lemmatizer = WordNetLemmatizer()
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            # print("desc",desc)
            desc = desc.split()
            # convert to lower case
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word) > 1]
            # remove tokens with numbers in them
            desc = [lemmatizer.lemmatize(word) for word in desc if word.isalpha()]
            # store as string
            desc = [word for word in desc if word != 'wa']
            desc_list[i] = ' '.join(desc)
            # doc = nlp(desc_list[i])
            # desc_list[i] = " ".join([token.lemma_ for token in doc])


# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
    # build a list of all description strings
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc


# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


# --------------------

def extract_features(directory):
    model = get_siamese_model((299, 299, 3))
    print(model.summary())
    features = []
    count = 0
    res = []
    # for name in listdir(directory):
    #   res.append(name)
    # res.sort()
    # print(res)
    # print("-----------------")
    for name in tqdm(range(300)):
        # load an image from file
        filename = directory + '/' + str(name) + '.PNG'
        # temp=name.split('.')
        filename2 = directory + '/' + str(name) + '_2.PNG'

        if os.path.exists(filename2) == False or os.path.exists(filename) == False:
            continue
        count += 1
        if count % 100 == 0:
            print(count)
            # if count==100:
            #   return features
        # print(filename," ",filename2)
        image1 = load_img(filename2, target_size=(299, 299))
        # convert the image pixels to a numpy array
        image1 = img_to_array(image1)
        # reshape data for the model
        image1 = image1.reshape((1, image1.shape[0], image1.shape[1], image1.shape[2]))
        image1 = preprocess_input(image1)
        imagesx1 = np.array(image1)
        imagesx1 = np.append(imagesx1, apply_affine_distortions(image1[0], 1)[0], axis=0)

        image2 = load_img(filename, target_size=(299, 299))
        # convert the image pixels to a numpy array
        image2 = img_to_array(image2)
        # reshape data for the model
        image2 = image2.reshape((1, image2.shape[0], image2.shape[1], image2.shape[2]))

        image2 = preprocess_input(image2)

        imagesx2 = np.array(image2)
        imagesx2 = np.append(imagesx2, apply_affine_distortions(image2[0], 1)[0], axis=0)
        # get features
        arf = []
        for i in range(imagesx1.shape[0]):
            img1 = np.asarray([imagesx1[i]])
            img2 = np.asarray([imagesx2[i]])
            inputs = [img1, img2]
            feature = model.predict(inputs, verbose=0)
            arf.append(feature)
        features.append(arf)

    return features


# -------------

# load doc into memory
def load_doc_file(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# load a pre-defined list of photo identifiers
def load_set(filename):
    brr = json.loads(open(filename).read())
    dataset = []
    for i in range(len(brr)):
        dataset.append(brr[i]['img_id'])

    return set(dataset)


# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
    # load document
    doc = load_doc_file(filename)
    descriptions = []
    for line in doc.split('\n'):
        # split line by white space
        # print("line:",line)
        tokens = line.split()
        # split id from description
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the set
        if image_id in dataset:
            # create list
            # wrap description in tokens
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            # store
            descriptions.append(desc)
    return descriptions


# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(descriptions)
    return tokenizer


# calculate the length of the description with the most words
def max_length(descriptions):
    return max(len(d.split()) for d in descriptions)


# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
    X1, X2, y = list(), list(), list()
    # walk through each image identifier
    for ii, desc in enumerate(descriptions):
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(photos[ii][0])
            X2.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(y)


# define the model
def define_model(vocab_size, max_length):
    # feature extractor model
    inputs1 = Input(shape=(128,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(512, activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 512, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(512)(se2)
    # decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(512, activation='relu',kernel_regularizer=l2(0.0001))(decoder1)
    outputs = Dense(vocab_size, activation='softmax',kernel_regularizer=l2(0.0001))(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize model
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    return model


# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text


def evaluate_model(model, descriptions, photos, tokenizer, max_length, index_img_test):
    actual, predicted = list(), list()
    # step over the whole set
    for i, desc in tqdm(enumerate(descriptions)):
        # generate description
        yhat = generate_desc(model, tokenizer, photos[i], max_length)
        actual.append(desc)
        predicted.append(yhat)
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

    for i in range(len(index_img_test)):
        print('Pair {0}\n actual: {1}\n predicted: {2}\n'.format(index_img_test[i], actual[i], predicted[i]))


def augment_descriptions(descriptions):
    ay_des = []
    wordNet, eda = text_init()
    for s in descriptions:
        ay_des.append(s)
        for i in range(5):
            ay_des.append(random_text_aug(wordNet, eda, s))

    d = defaultdict(lambda: 0)
    for s in ay_des:
        for w in s.split():
            d[w] += 1

    for i in range(len(ay_des)):
        for w in ay_des[i].split():
            if d[w] < 3:
                ay_des[i] = ay_des[i].replace(w, '')
    return ay_des


if __name__ == "__main__":
    model = get_siamese_model((299, 299, 3))
    print(model.summary())

    # load descriptions
    descriptions = load_doc()
    print(descriptions['1'])
    print('Loaded: %d ' % len(descriptions))
    # clean descriptions
    clean_descriptions(descriptions)
    # summarize vocabulary
    vocabulary = to_vocabulary(descriptions)
    print('Vocabulary Size: %d' % len(vocabulary))
    # save to file
    save_descriptions(descriptions, 'descriptions.txt')

    directory = 'myhouse_dataset/myhouse_dataset'
    features = extract_features(directory)

    filename = 'myhouse_dataset/annotations.json'
    train = load_set(filename)
    print('Dataset: %d' % len(train))
    # descriptions
    descriptions = load_clean_descriptions('descriptions.txt', train)
    index_img = [i for i in range(len(descriptions))]
    # prepare sequences
    (descriptions_train, descriptions_test, ar_train, ar_test, index_img_train, index_img_test) = train_test_split(
        descriptions, features, index_img, test_size=0.25)

    features_train = []
    for af in ar_train:
        for f in af:
            features_train.append(f)

    features_test = []
    for af in ar_test:
        features_test.append(af[0])

    descriptions_train = augment_descriptions(descriptions_train)
    features_train, descriptions_train = shuffle(features_train, descriptions_train, random_state=14)

    print('Descriptions: train=%d' % len(descriptions_train))
    # photo features
    print('Photos: train=%d' % len(features_train))
    # prepare tokenizer
    tokenizer = create_tokenizer(descriptions_train + descriptions_test)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)
    # determine the maximum sequence length
    max_length = max_length(descriptions_train + descriptions_test)
    print('Description Length: %d' % max_length)

    data = '\n'.join(descriptions_train)
    file = open('descriptions-ay.txt', 'w')
    file.write(data)
    file.close()

    Ximage_train1, Ximage_train2, y_train = create_sequences(tokenizer, max_length, descriptions_train, features_train,
                                                             vocab_size)
    Ximage_test1, Ximage_test2, y_test = create_sequences(tokenizer, max_length, descriptions_test, features_test,
                                                          vocab_size)

    model = define_model(vocab_size, max_length)
    filepath = 'neural-network-models/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    model.fit([Ximage_train1, Ximage_train2], y_train, epochs=40, verbose=True,  # callbacks=[checkpoint],
              validation_data=([Ximage_test1, Ximage_test2], y_test))

    evaluate_model(model, descriptions_test, features_test, tokenizer, max_length, index_img_test)

    model.save_weights("final_siam" + str(time.time()) + ".h5")
