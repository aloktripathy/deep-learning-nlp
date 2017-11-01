import numpy as np

import os
import sys

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)


def indexes_to_words(word_index_sequence, w2v_model):
    words = []
    for idx in word_index_sequence:
        words.append(w2v_model.wv.index2word[idx])
    return ' '.join(words)


def temperature_sample(a, temperature):
    return a.argmax()
    a[a==0] = 1e-20
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    # We are doing this to deal with problem caused by rounding where sum of a is not 1.
    a = a.astype(np.float64)
    a = a / sum(a)
    a[a.argmax()] += 1 - a.sum()

    return np.argmax(np.random.multinomial(1, a, 1))
