import unicodedata
import re
import numpy
from numpy import eye, argmax, array

DEFAULT_CHAR = u' '
CHARS_LOWER = u'abcdefghijklmnopqrstuvwxyzáéíóöúüőű'
CHARS_UPPER = u'ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÖÚÜŐŰ'
CHARS_LOWER_NO_ACCENT = u'abcdefghijklmnopqrstuvwxyzaeioouuou'

INTEREST_SET = u'aeiouAEIOUáéíóöúüőűÁÉÍÓÖÚÜŐŰ'

FILTER_SET = {c for c in CHARS_UPPER + CHARS_LOWER}

UPPER_ENCODER_MAP = {c: (cn, 0 if c == cn else 1) for c, cn in zip(CHARS_UPPER, CHARS_LOWER)}
UPPER_DECODER_MAP = dict((v, k) for k, v in UPPER_ENCODER_MAP.items())

ACCENT_ENCODER_MAP = {
    'á': ('a', 1),
    'é': ('e', 1),
    'í': ('i', 1),
    'ó': ('o', 1),
    'ö': ('o', 2),
    'ő': ('o', 3),
    'ú': ('u', 1),
    'ü': ('u', 2),
    'ű': ('u', 3)
}

ACCENT_DECODER_MAP = dict((v, k) for k, v in ACCENT_ENCODER_MAP.items())


def unzip(pairs):
    return tuple([list(t) for t in zip(*pairs)])


def group(text, n, m):
    return [text[i:i + n].ljust(n) for i in range(0, len(text), m)]


def flatten(list):
    return [item for sublist in list for item in sublist]


class Filter:
    def __init__(self, filterSet, defaultElement):
        self.filterSet = filterSet
        self.defaultElement = defaultElement

    def _encode(self, item):
        return (item, 0) if item in self.filterSet else (self.defaultElement, 1)

    def _decode(self, triple):
        (c, i, p) = triple
        return p if (i == 1) else c

    def encode(self, items):
        return unzip([self._encode(item) for item in items])

    def decode(self, items, masks, originalItems):
        return [self._decode(item) for item in zip(items, masks, originalItems)]


class Coder:
    def __init__(self, encoderMap, decoderMap):
        self.encoderMap = encoderMap
        self.decoderMap = decoderMap

    def _encode(self, item):
        return self.encoderMap.get(item, (item, 0))

    def _decode(self, pair):
        return self.decoderMap.get(pair, pair[0])

    def encode(self, items):
        return unzip([self._encode(item) for item in items])

    def decode(self, items, masks):
        return [self._decode(item) for item in zip(items, masks)]


class OneHotIndex:
    def __init__(self, n):
        self.n = n
        self.encoderMap = [[(1 if i == j else 0) for j in range(n)] for i in range(n)]

    def _encode(self, index):
        return self.encoderMap[index]

    def _decode(self, onehot):
        return max(range(len(onehot)), key=onehot.__getitem__)

    def encode(self, many):
        return [self._encode(item) for item in many]

    def decode(self, many):
        return [self._decode(item) for item in many]


class OneHot:
    def __init__(self, values):
        values = list(set(values))
        self.mapping = dict((c, i) for i, c in enumerate(values))
        self.reverse_mapping = dict((i, c) for i, c in enumerate(values))

        self.n = len(values)

        self.encoderMap = [[(1 if i == j else 0) for j in range(self.n)] for i in range(self.n)]

    def _encode(self, value):
        return self.encoderMap[self.mapping.get(value)]

    def _decode(self, onehot):
        return self.reverse_mapping.get(max(range(len(onehot)), key=onehot.__getitem__))

    def encode(self, many):
        return [self._encode(item) for item in many]

    def decode(self, many):
        return [self._decode(item) for item in many]


onehot4 = OneHotIndex(4)
onehot = OneHot(CHARS_LOWER_NO_ACCENT + DEFAULT_CHAR)
coderFilter = Filter(FILTER_SET, DEFAULT_CHAR)
coderUpper = Coder(UPPER_ENCODER_MAP, UPPER_DECODER_MAP)
coderAccent = Coder(ACCENT_ENCODER_MAP, ACCENT_DECODER_MAP)


def encoder(text):
    (filtered_text, filter_mask) = coderFilter.encode(text)
    (lowered_text, lowered_mask) = coderUpper.encode(filtered_text)
    (unaccented_text, unaccented_mask) = coderAccent.encode(lowered_text)

    unaccented_mask_onehot = onehot4.encode(unaccented_mask)
    unaccented_text_onehot = onehot.encode(unaccented_text)

    return unaccented_text_onehot, unaccented_mask_onehot, unaccented_text, lowered_mask, filter_mask, text


def decoder(unaccented_text_onehot, unaccented_mask_onehot, unaccented_text, lowered_mask, filter_mask, text):
    # unaccented_text = onehot.decode(unaccented_text_onehot)
    unaccented_mask = onehot4.decode(unaccented_mask_onehot)

    accented_text = coderAccent.decode(unaccented_text, unaccented_mask)
    uppered_text = coderUpper.decode(accented_text, lowered_mask)
    restored_text = coderFilter.decode(uppered_text, filter_mask, text)

    return restored_text


def encoder_group(text, n, m):
    text_with_accent_groups = group(text, n, m)
    return unzip([encoder(t) for t in text_with_accent_groups])


def decoder_group(unaccented_text_onehot_groups, unaccented_mask_onehot_groups, lowered_mask_groups, filter_mask_groups,
                  text_groups):
    return [decoder(*g) for g in
            zip(unaccented_text_onehot_groups, unaccented_mask_onehot_groups, lowered_mask_groups, filter_mask_groups,
                text_groups)]


from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Reshape
from keras.optimizers import SGD
from keras.models import Model, load_model

from keras import backend as K
import os
import importlib

def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        importlib.reload(K)
        assert K.backend() == backend

#set_keras_backend("theano")
from time import time

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend import manual_variable_initialization
#tf.keras.backend.manual_variable_initialization(True)

class Trainer:

    def __init__(self, window_size):

        self.window_size = window_size
        inputs = tf.keras.layers.Input(shape=(window_size, onehot.n), name='input')
        flatten = tf.keras.layers.Flatten(name='flatten')(inputs)
        layer = tf.keras.layers.Dense(units=100, activation='relu', name='hidden1')(flatten)
        #layer = Dense(units=100, activation='relu', name='hidden2')(layer)
        outputs = tf.keras.layers.Dense(units=onehot4.n, activation='sigmoid', name='output')(layer)
        # outputs = Reshape((window_size, onehot4.n))(outputs)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        w1 = self.model.get_weights()

        #self.model = tf.keras.models.load_model('m_a4.h5')
        self.model.load_weights('w_a4.h5', by_name=True)
        w2 = self.model.get_weights()


        #self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        for a, b in zip(w1, w2):
            if numpy.all(a == b):
                print ("wtf is happening")

        #tf.keras.Model(inputs=inputs, outputs=outputs )
        #self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        #self.model.load_weights('w_a4.h5')

    def windows(self, original_text, window_size):

        window_size = int((window_size - 1)/2)
        text = ' ' * window_size + original_text + ' ' * window_size

        window = [(encoder(text[i - window_size:i + 1 + window_size]), i - window_size) for i, c in enumerate(text) if
                  c in INTEREST_SET]

        x = [e[0] for (e, i) in window]
        y = [e[1][window_size] for (e, i) in window]
        r = [(e[2][window_size], e[3][window_size], e[4][window_size], e[5][window_size]) for (e, i) in window]
        i = [i for (e, i) in window]

        return x, y, r, i

    def restore(self, original_text, y, r, i):
        text = list(original_text)
        for (yi, ri, ii) in zip(y, r, i):
            unaccented_mask = onehot4._decode(yi)
            unaccented_char, lowered_mask, filter_mask, char = ri

            accented_char = coderAccent._decode((unaccented_char, unaccented_mask))
            uppered_char = coderUpper._decode((accented_char, lowered_mask))
            restored_char = coderFilter._decode((uppered_char, filter_mask, char))

            text[ii] = restored_char

        return ''.join(text)

    def train(self, text):

        x_train, y_train, *restore = self.windows(text, self.window_size)

        x_train = array(x_train)
        y_train = array(y_train)

        sgd = tf.keras.optimizers.SGD(lr=0.01)
        self.model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
        #
        # print(self.model.summary())

        tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time()))

        #
        # # train only once (epochs=1) over the whole x_train/y_train
        self.history = self.model.fit(x_train, y_train, verbose=0, epochs=3, validation_split=0.4, callbacks=[tensorboard])

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.history = self.model.fit(x_train, y_train, batch_size=1000, verbose=0, epochs=3, validation_split=0.4, callbacks=[tensorboard])
        #
        #self.model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
        #self.history = self.model.fit(x_train, y_train, batch_size=10000, verbose=1, epochs=10, validation_split=0.0)

        self.model.save_weights('w_a4.h5')
        self.model.save('m_a4.h5')

        # decoded = decoder_group(x_train, y_train, *restore)

        # print(''.join(flatten(decoded)))

    def load(self):
        w1 = self.model.get_weights()

        #self.model.load_weights('w_a4.h5')

        w2 = self.model.get_weights()


        #self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        for a, b in zip(w1, w2):
            if numpy.all(a == b):
                print ("wtf is happening")

    def correct(self, text):
        x_predict, y_predict_dummy, *restore = self.windows(text, self.window_size)

        x_predict = array(x_predict)

        y_predict = self.model.predict(x_predict)
        # print(y_predict)

        decoded = self.restore(text, y_predict, *restore)
        # print(decoded)

        return ''.join(decoded)

    def plot(self):
        print(self.history.history.keys())

        loss_list = [s for s in self.history.history.keys() if 'loss' in s and 'val' not in s]
        val_loss_list = [s for s in self.history.history.keys() if 'loss' in s and 'val' in s]
        acc_list = [s for s in self.history.history.keys() if 'acc' in s and 'val' not in s]
        val_acc_list = [s for s in self.history.history.keys() if 'acc' in s and 'val' in s]

        if len(loss_list) == 0:
            print('Loss is missing in history')
            return

        ## As loss always exists
        epochs = range(1, len(self.history.history[loss_list[0]]) + 1)

        ## Loss
        plt.figure(1)
        for l in loss_list:
            plt.plot(epochs, self.history.history[l], 'b',
                     label='Training loss (' + str(str(format(self.history.history[l][-1], '.5f')) + ')'))
        for l in val_loss_list:
            plt.plot(epochs, self.history.history[l], 'g',
                     label='Validation loss (' + str(str(format(self.history.history[l][-1], '.5f')) + ')'))

        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        ## Accuracy
        plt.figure(2)
        for l in acc_list:
            plt.plot(epochs, self.history.history[l], 'b',
                     label='Training accuracy (' + str(format(self.history.history[l][-1], '.5f')) + ')')
        for l in val_acc_list:
            plt.plot(epochs, self.history.history[l], 'g',
                     label='Validation accuracy (' + str(format(self.history.history[l][-1], '.5f')) + ')')

        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()


trainer = Trainer(5)

with open('hu.txt', 'r') as myfile: accented_text = myfile.read().replace('\n', '')
print (len(accented_text))
accented_text = accented_text[1:600000]
trainer.train(accented_text)
#trainer.train(accented_text)
trainer.plot()

text = 'MAgányosan, törpe fák alatt!';


#trainer.load()
print(trainer.correct('magányosan, törpe fák alatt!'))

# print(groups)
# x_train, y_train, *_  = groups
# decoded = decoder_group(*groups)
# print(''.join(flatten(decoded)))

# accent_mask = unaccented_mask
# accent_mask[4] = 0
#
#
# print(filtered_text)
# print(lowered_text)
# print(unaccented_text)
# print(onehot4.encode(unaccented_mask))
# print(onehot.decode(onehot.encode(unaccented_text)))
#
# print(accented_text)
# print(uppered_text)
# print(restored_text)

# chars_norm = unicodedata.normalize('NFD', chars).encode('ascii', 'ignore').decode('ascii')
# chars_lower_norm = unicodedata.normalize('NFD', chars_lower).encode('ascii', 'ignore').decode('ascii')
#
# CHAR_MAP = {c: cn for c, cn in zip(chars, chars_norm)}
# CHAR_MAP_TO_LOWER = {c: cn for c, cn in zip(chars_upper, chars_lower)}
#
# accented_text = u"ez egy szöveg."
# with open('valid.txt', 'r') as myfile: accented_text = myfile.read().replace('\n', '')
#
# accented_text = ''.join([CHAR_MAP_TO_LOWER.get(c, c) for c in accented_text])
#
# unaccented_text = ''.join([CHAR_MAP.get(c, '_') for c in accented_text])
#
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
# from keras.layers import Input, Dense, Dropout, Activation, Flatten, Reshape
# from keras.optimizers import SGD
# from keras.models import Model, load_model
#
#
# class CharacterMapping:
#
#     def __init__(self, chars):
#         self.alphabet = sorted(list(set(chars)))
#         self.mapping = dict((c, i + 1) for i, c in enumerate(self.alphabet))
#         self.reverse_mapping = dict((i + 1, c) for i, c in enumerate(self.alphabet))
#
#         self.n = len(self.alphabet) + 1
#         u = eye(self.n, self.n)
#         self.onehot = dict((i, u[i]) for i in range(self.n))
#
#     def text2seq(self, text):
#         return [self.mapping.get(char, 0) for char in text]
#
#     def seq2text(self, seq):
#         return ''.join([self.reverse_mapping.get(index, ' ') for index in seq])
#
#     def seq2onehot(self, seq):
#         return array([self.onehot[index] for index in seq])
#
#     def text_array_to_seq_array(self, text_array, padding=None):
#         seq_array = [self.text2seq(text) for text in text_array]
#         if padding is not None:
#             return pad_sequences(seq_array, maxlen=padding, padding='post', value=0)
#         return seq_array
#
#     def seq_array_to_text_array(self, seq_array):
#         return [self.seq2text(seq) for seq in seq_array]
#
#     def seq_array_to_onehot_array(self, seq_array):
#         return [self.seq2onehot(seq) for seq in seq_array]
#
#     def onehot_array_to_seq_array(self, onehot_array):
#         return argmax(onehot_array, axis=2)
#
#     def text_array_to_onehot_array(self, text_array, padding=None):
#         return self.seq_array_to_onehot_array(self.text_array_to_seq_array(text_array, padding))
#
#     def onehot_array_to_text_array(self, onehot_array):
#         return self.seq_array_to_text_array(self.onehot_array_to_seq_array(onehot_array))
#
#     def group(selfself, text, n):
#         return [text[i:i + n].ljust(n) for i in range(0, len(text), n)]
#
#
# # print(chars_lower)
# cm = CharacterMapping(chars_lower)
# # print('=====')
# accented_words = cm.group(accented_text, 20)
# unaccented_words = cm.group(unaccented_text, 20)
# x_train = array(cm.text_array_to_onehot_array(unaccented_words))
# y_train = array(cm.text_array_to_onehot_array(accented_words))
#
# # print(x_train)
# # print(y_train)
#
# # print(cm.onehot_array_to_text_array(x_train))
#
#
# inputs = Input(shape=(20, cm.n), name='input')
# flatten = Flatten(name='flatten')(inputs)
# layer = Dense(units=400, activation='relu', name='hidden')(flatten)
# outputs = Dense(units=20 * cm.n, activation='sigmoid', name='output')(layer)
# outputs = Reshape((20, cm.n))(outputs)
#
# model = Model(inputs=inputs, outputs=outputs)
#
# sgd = SGD(lr=0.01)
# model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
#
# print(model.summary())
#
# # train only once (epochs=1) over the whole x_train/y_train
# history = model.fit(x_train, y_train, verbose=1, epochs=5, validation_split=0.4)
#
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# history = model.fit(x_train, y_train, verbose=1, epochs=100, validation_split=0.4)
#
# model.save('model_accenter2.h5')
#
# unaccented_words2 = cm.group('ez egy szoveg.', 20)
# x_test = array(cm.text_array_to_onehot_array(unaccented_words2))
#
# y_test = model.predict(x_test)
#
# print(cm.onehot_array_to_text_array(y_test))
#
# # xte,yte = test.training_data(2)
# # xte = array(cm.text_array_to_onehot_array(xte, padding=30))
#
# # print(model.predict(xte))
