import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import models, layers, optimizers
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import bz2
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import re
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import RNN
from tensorflow.python.keras.layers import GRU
import pickle
import sys

import os


def get_dataset():
    def get_labels_and_texts(file):
        labels = []
        texts = []
        for line in bz2.BZ2File(file):
            x = line.decode("utf-8")
            labels.append(int(x[9]) - 1)
            texts.append(x[10:].strip())
        return np.array(labels), texts

    if False:
        # Amazon product reviews
        train_labels, train_texts = get_labels_and_texts('/Users/raananf/Downloads/test.ft.txt.bz2')
    else:
        # IMDB movie reviews
        movie_reviews = pd.read_csv("IMDB Dataset.csv")
        # movie_reviews = pd.read_csv("/cs/dataset/IMDB_reviews/IMDB Dataset.csv")
        train_texts = movie_reviews['review']
        sent = movie_reviews['sentiment']
        train_labels = list(map(lambda x: 1 if x == "positive" else 0, sent))

    print("Dataset size: %d" % len(train_labels))

    p = np.random.permutation(len(train_texts))
    train_texts = [train_texts[p[i]] for i in range(len(train_texts))]
    train_labels = [train_labels[p[i]] for i in range(len(train_labels))]

    # Normlaize

    TAG_RE = re.compile(r'<[^>]+>')

    def remove_tags(text):
        return TAG_RE.sub('', text)

    def preprocess_text(sen):
        # Removing html tags
        sentence = remove_tags(sen)

        # Remove punctuations and numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)

        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

        # Removing multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)

        return sentence

    print("Normalizing Dataset")

    NON_ALPHANUM = re.compile(r'[\W]')
    NON_ASCII = re.compile(r'[^a-z0-1\s]')

    test_ascii = []

    def normalize_texts(texts, labels):
        normalized_texts = []
        normalized_labels = []
        for i in range(len(texts)):
            text = texts[i]
            label = labels[i]

            lower = text.lower()

            lower = preprocess_text(lower)

            test_ascii.append(lower.split(" "))

            normalized_texts.append(lower)
            normalized_labels.append(label)
            continue

            no_punctuation = NON_ALPHANUM.sub(r' ', lower)
            no_non_ascii = NON_ASCII.sub(r'', no_punctuation)
            normalized_texts.append(no_non_ascii)
        return normalized_texts, normalized_labels

    train_texts, train_labels = normalize_texts(train_texts, train_labels)

    train_labels = np.asarray(train_labels)

    # validation
    print("Test Set")

    MAX_FEATURES = 5000
    # print("MAX_FEATURES: %d" % MAX_FEATURES)
    tokenizer = Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(train_texts)
    num_words = len(tokenizer.word_index.items())
    MAX_FEATURES = num_words + 1

    test_texts = train_texts[:500]
    test_labels = train_labels[:500]
    train_texts = train_texts[500:]
    train_labels = train_labels[500:]
    test_ascii = test_ascii[:500]

    ##########################
    # ADD YOUR OWN TEST TEXT #
    ##########################

    my_test_texts = []
    my_test_texts.append("great movie if you like to suffer")
    my_test_texts.append("i thought that it will be a waste of time but the movie was perfect")
    my_test_texts.append("huge waste of time")

    ##########################
    ##########################

    for k in range(len(my_test_texts)):
        test_texts[k] = my_test_texts[k]
        test_ascii[k] = my_test_texts[k].split(" ")

    train_texts = tokenizer.texts_to_sequences(train_texts)
    test_texts = tokenizer.texts_to_sequences(test_texts)

    # pad
    print("Padding")
    MAX_LENGTH = 100  # max(len(train_ex) for train_ex in train_texts)
    print("Max sentence length: %d" % MAX_LENGTH)
    train_texts = pad_sequences(train_texts, padding='post', maxlen=MAX_LENGTH)
    test_texts = pad_sequences(test_texts, padding='post', maxlen=MAX_LENGTH)

    # read embedding mat

    print("Read embedding matrix")
    embeddings_dictionary = dict()
    glove_file = open('glove.6B.100d.txt', encoding="utf8")
    # glove_file = open('/cs/dataset/IMDB_reviews/glove.6B.100d.txt', encoding="utf8")

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()

    embedding_matrix = np.zeros((num_words + 1, 100))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    return train_texts, train_labels, test_texts, test_labels, test_ascii, embedding_matrix, MAX_LENGTH, MAX_FEATURES
