##########################
# Code for Ex. #2 in IDL #
##########################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import models, layers, optimizers
import tensorflow as tf
import bz2
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import keras.backend as K
import sys
from tensorflow.keras.activations import sigmoid, relu, tanh, softmax
from collections import defaultdict
import os
import loader as ld
import argparse
import optuna
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
import matplotlib.ticker as ticker

sys.setrecursionlimit(2500)

train_texts, train_labels, test_texts, test_labels, test_ascii, embedding_matrix, MAX_LENGTH, MAX_FEATURES = ld.get_dataset()

#####################
# Execusion options #
#####################

TRAIN = False

RECR = False  # recurrent netowrk (RNN/GRU) or a non-recurrent network

ATTN = True  # use attention layer in global sum pooling or not
LSTM = False  # use LSTM or otherwise RNN
WEIGHTED = True
COMPARE_MODELS = False


def set_style():
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.facecolor'] = '00000007'
    plt.rcParams['axes.edgecolor'] = '0000003D'
    plt.rcParams['axes.labelcolor'] = '000000D9'
    plt.rcParams['xtick.color'] = '000000'
    plt.rcParams['ytick.color'] = '000000'
    plt.rcParams['legend.facecolor'] = 'FFFFFFD9'
    plt.rcParams['legend.edgecolor'] = '000000D9'
    plt.rcParams['figure.facecolor'] = 'FFFFFF'
    plt.rcParams['savefig.facecolor'] = 'FFFFFF'

    plt.rcParams['figure.figsize'] = 12, 8


# Getting activations from model

def get_act(net, input, name):
    sub_score = [layer for layer in net.layers if name in layer.name][0].output
    # functor = K.function([test_texts]+ [K.learning_phase()], sub_score)

    OutFunc = K.function([net.input], [sub_score])
    return OutFunc([test_texts])[0]


def f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


# RNN Cell Code

def RNN(dim, x):
    # Learnable weights in the cell
    Wh = layers.Dense(dim, use_bias=False)
    Wx = layers.Dense(dim)

    # unstacking the time axis
    x = tf.unstack(x, axis=1)

    H = []

    h = tf.zeros_like(Wx(x[0]))

    for i in range(len(x)):
        # Apply the basic step in each time step

        # -- missing code --

        h = relu(Wh(h) + Wx(x[i]))

        H.append(h)

    H = tf.stack(H, axis=1)

    return h, H


# GRU Cell Code

def GRU(dim, x):
    # Learnable weights in the cell
    Wzx = layers.Dense(dim)
    Wzh = layers.Dense(dim, use_bias=False)

    Wrx = layers.Dense(dim)
    Wrh = layers.Dense(dim, use_bias=False)

    Wx = layers.Dense(dim)
    Wh = layers.Dense(dim, use_bias=False)

    # unstacking the time axis
    x = tf.unstack(x, axis=1)

    H = []

    h = tf.zeros_like(Wx(x[0]))

    for i in range(len(x)):
        # -- missing code --
        z = sigmoid(Wzx(x[i]) + Wzh(h))
        r = sigmoid(Wrx(x[i]) + Wrh(h))
        ht = tanh(Wx(x[i]) + Wh(h) * r)
        h = (1 - z) * h + z * ht

        H.append(h)

    H = tf.stack(H, axis=1)

    return h, H


# (Spatially-)Restricted Attention Layer
# k - specifies the -k,+k neighbouring words

def restricted_attention(x, k):
    dim = x.shape[2]

    Wq = layers.Dense(dim)
    Wk = layers.Dense(dim)

    wk = Wk(x)

    paddings = tf.constant([[0, 0, ], [k, k], [0, 0]])
    pk = tf.pad(wk, paddings)
    pv = tf.pad(x, paddings)

    keys = []
    vals = []
    for i in range(-k, k + 1):
        keys.append(tf.roll(pk, i, 1))
        vals.append(tf.roll(pv, i, 1))

    keys = tf.stack(keys, 2)
    keys = keys[:, k:-k, :, :]
    vals = tf.stack(vals, 2)
    vals = vals[:, k:-k, :, :]

    # -- missing code --
    query = Wq(x)[..., None]

    dot_product = tf.matmul(keys, query) / np.sqrt(dim)
    atten_weights = layers.Softmax(name='atten_weights', axis=-2)(dot_product)

    val_out = tf.matmul(atten_weights, vals, transpose_a=True)
    val_out = tf.squeeze(val_out, axis=2)
    return x + val_out


# Building Entire Model
def build_model(model_type, n_unints=64):
    print(model_type)
    sequences = layers.Input(shape=(MAX_LENGTH,))
    embedding_layer = layers.Embedding(MAX_FEATURES, 100, weights=[embedding_matrix], input_length=MAX_LENGTH,
                                       trainable=False)

    # embedding the words into 100 dim vectors

    x = embedding_layer(sequences)

    if model_type not in {'RNN', 'GRU'}:

        # non recurrent networks

        if model_type in {'ATTN_WEIGHTED', 'ATTN_SUM'}:
            # attention layer
            x = restricted_attention(x, k=5)

        # word-wise FC layers -- MAKE SURE you have ,name= "sub_score" in the sub_scores step
        # E.g., sub_score = layers.Dense(2,name="sub_score")(x)

        # -- missing code --
        x = layers.Dense(32, activation='relu')(x)
        # x = layers.Dense(50, activation='relu')(x)

        if model_type in {'WEIGHTED', 'ATTN_WEIGHTED'}:
            x = layers.Dense(2, name="sub_score")(x)
            x0 = layers.Lambda(lambda x: x[:, :, 0])(x)
            x1 = layers.Lambda(lambda x: x[:, :, 1])(x)
            sum_weights = layers.Softmax(name='sum_weights')(x1)
            x = tf.expand_dims(x0 * sum_weights, 2)
        else:
            x = layers.Dense(1, name="sub_score")(x)
        x = K.sum(x, axis=1)

        # final prediction

        x = tf.sigmoid(x)

        predictions = x

    else:
        # recurrent networks
        if model_type == 'GRU':
            x, _ = GRU(n_unints, x)
        else:

            x, _ = RNN(n_unints, x)

        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dense(1, activation='sigmoid')(x)

        predictions = x

    model = models.Model(inputs=sequences, outputs=predictions)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy', f1]
    )
    return model


def optimize_rnn_gru(trial):
    assert TRAIN and RECR
    n_uints = trial.suggest_int("n_units", 64, 128)
    print(n_uints)
    model = build_model(n_uints)
    return train_model(model, '')


def train_model(model, checkpoint_path):
    print("Training")
    callbacks = []
    if checkpoint_path:
        os.path.dirname(checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True)
        callbacks.append(cp_callback)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    callbacks.append(early_stopping)

    # print(model.summary())
    history = model.fit(
        train_texts,
        train_labels,
        batch_size=128,
        epochs=25,
        validation_data=(test_texts, test_labels), callbacks=callbacks)
    return max(history.history['val_binary_accuracy'])


def compare_result_test(models):
    d = defaultdict(list)
    for name, model in models.items():
        model_results = model.evaluate(test_texts, test_labels, batch_size=128, verbose=0)
        for k, metric in enumerate(model.metrics_names):
            d[metric].append(model_results[k])
        # plot_cm(model.predict(test_texts), test_labels, print_results=False, model_name=name)

    colors = ['#780707', '#557f2d', '#2d7f5e']
    df = pd.DataFrame(d, index=[name for name in models.keys()])
    ax = df.plot(kind='bar', color=colors, width=.8)
    for p in ax.patches:
        ax.annotate(np.round(p.get_height(), decimals=2), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                    va='center', xytext=(0, 10), textcoords='offset points')
    plt.title('Test Results')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()


def print_result(model, model_name):
    print("Model Name: ", model_name)

    print("Example Predictions:")
    preds = model.predict(test_texts)
    if model_name not in {'GRU', 'RNN'}:

        sub_score = get_act(model, test_texts, "sub_score")
        if 'ATTN' in model_name:
            att_weights = get_act(model, test_texts, 'atten_weights')
        if 'WEIGHTED' in model_name:
            weights = get_act(model, test_texts, 'sum_weights')

    for i in range(1, 2):

        print("-" * 20)

        if model_name not in {'GRU', 'RNN'}:
            idx = ['Score']

            if 'ATTN' in model_name:
                plot_attention(np.squeeze(att_weights[i]), test_ascii[i])
            if 'WEIGHTED' in model_name:
                idx.append('Weight')
                f, a = plt.subplots(1, 2)

                w = np.squeeze(K.softmax(sub_score[i, :, 1]))
                s = np.squeeze(sub_score[i, :, 0])
                df1 = pd.DataFrame.from_records(list(zip(test_ascii[i], s)), columns=['Words', 'Scores']).set_index(
                    'Words')
                df2 = pd.DataFrame.from_records(list(zip(test_ascii[i], w)), columns=['Words', 'Weights']).set_index(
                    'Words')
                df1.plot(kind='bar', ax=a[0])
                df2.plot(kind='bar', ax=a[1], color='orange')

                a[0].set_title('Scores')
                a[1].set_title('Weights')


            else:
                df = pd.DataFrame.from_records(list(zip(test_ascii[i], np.squeeze(sub_score[i]))),
                                               columns=['Words', 'Scores'])
                df = df.set_index('Words')
                df.plot(kind='bar')
            plt.suptitle('Model: {:s}\n Prediction: {:.3f}'.format(model_name, *preds[i]))

            plt.show()
            # print words along with their sub_score

            num = min((len(test_ascii[i]), 100))
            print(test_ascii[i])

            for k in range(num):

                # -- missing code --
                if 'SUM' in model_name:
                    print(test_ascii[i][k], *sub_score[i, k])

                elif 'WEIGHTED' in model_name:
                    print(test_ascii[i][k], '\nScore: ', sub_score[i, k][0], ' Weight: ', weights[i, k])

            print("\n")
        else:
            print(test_ascii[i])
            print(preds[i])
        if preds[i] > 0.5:
            print('Prediction: ', "Positive")
        else:
            print('Prediction: ', "Negative")

        print("-" * 20)

    print('Accuracy score: {:0.4}'.format(accuracy_score(test_labels, 1 * (preds > 0.5))))
    print('F1 score: {:0.4}'.format(f1_score(test_labels, 1 * (preds > 0.5))))
    print('ROC AUC score: {:0.4}'.format(roc_auc_score(test_labels, preds)))


def plot_attention(attention, sentence):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    attention = attention[:len(sentence), :len(sentence)]

    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-mode", type=str, default='train', dest='mode',
                        help='Train a model or test a trained one')

    parser.add_argument("-model", type=str, default='GRU', dest='model',
                        help='options:RNN, GRU, SUM, WEIGHTED,ATTN_WEIGHTED,ATTN_SUM')

    parser.add_argument("-to_compare", action='store_true', default=False, dest='to_compare',
                        help='Compare the saved models results or not')

    # Recurrent/GRU parameters
    parser.add_argument("-rnn_units", type=int, default=86, dest='rnn_units',
                        help='Number of hidden layers')
    parser.add_argument("-gru_units", type=int, default=107, dest='gru_units',
                        help='Number of hidden layers')
    args = parser.parse_args(['-to_compare'])

    return args


def main():
    args = get_args()
    model_name = args.model

    if not args.to_compare:

        checkpoint_path = 'model_save/{:s}.ckpt'.format(model_name)
        model = build_model(model_name)
        if args.mode == 'train':
            train_model(model=model, checkpoint_path=checkpoint_path)
        else:
            model.load_weights(checkpoint_path).expect_partial()
        print_result(model, model_name)

    else:

        models_params = {'RNN': args.rnn_units, 'GRU': args.gru_units, 'SUM': 0, 'WEIGHTED': 0,
                         'ATTN_WEIGHTED': 0, 'ATTN_SUM': 0}

        models_dict = defaultdict(list)

        for model_name, n_unit in models_params.items():
            model = build_model(model_name, n_unit)
            model_path = 'model_save/{:s}.ckpt'.format(model_name)
            model.load_weights(model_path).expect_partial()
            models_dict[model_name] = model
            print_result(model, model_name)

        compare_result_test(models_dict)


if __name__ == '__main__':
    set_style()
    main()
