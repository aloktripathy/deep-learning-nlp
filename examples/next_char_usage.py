import os
import re
import sys
import time

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

from rnn.next_char import NextChar

import tensorflow as tf
import numpy as np
from tflearn.data_utils import string_to_semi_redundant_sequences
from tflearn.data_utils import chars_to_dictionary

with open('../data/shakespear/main/romeo_and_juliet.txt') as fp:
    txt = ''.join(fp.readlines())


def get_data_stream(batch_size, char_dict, seq_len):
    x, y, char_idx = string_to_semi_redundant_sequences(txt, seq_maxlen=seq_len, char_idx=char_dict)
    batch_count = x.shape[0] // batch_size

    def iter():
        for i in range(batch_count):
            idx = i*batch_size
            x_batch = x[idx:idx+batch_size, :, :].argmax(2)
            targets = y[idx:idx+batch_size, :].argmax(1)

            y_batch = np.zeros(x_batch.shape, dtype=np.uint8)
            y_batch[:, :-1] = x_batch[:, 1:]
            y_batch[:, -1] = targets
            yield x_batch, y_batch
    return iter


def index_to_chars(int_sequence, idx_dict):
    inv_dict = dict(zip(idx_dict.values(), idx_dict.keys()))
    return ''.join([inv_dict[int_v] for int_v in int_sequence])


def char_to_index(char_sequence, idx_dict):
    return [idx_dict[char] for char in char_sequence]


def _get_log_summary_writer(graph, name, version):
    path = "/tmp/{}/{}".format(name, version)
    return tf.summary.FileWriter(path, graph=graph)


def get_seed(seq_size):
    global start_indexes
    if not start_indexes:
        start_indexes = [match.start() for match in re.finditer(re.escape('\n'), txt)]
    start_index = np.random.choice(start_indexes) + 1
    return txt[start_index: start_index+seq_size]


start_indexes = None
rnn_size = 256
batch_size = 64
n_layers = 3
epochs = 50
sequence_length = 25
lr = 0.01

char_idx = chars_to_dictionary(txt)
alphabet_size = len(char_idx)

model = NextChar(alphabet_size, rnn_size, n_layers, batch_size)
model.setup_training()

session = tf.Session()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()

writer = _get_log_summary_writer(session.graph, 'next-char-b1', 1)

s = model.restore_model(
    session,
    '/home/alok/Desktop/deep-learning/nlp_cs224n/checkpoints/shakespeare/',
    '1-50.meta'
)
if s:
    saver = s

loss = None
data_stream = get_data_stream(batch_size, char_idx, seq_len=sequence_length)

for e in range(epochs):
    current_state = None
    t = -time.time()
    for x, y in data_stream():
        current_state, loss = model.train(session, x, y, current_state)
    t += time.time()
    print('*' * 100)
    print('Epoch', e+1, 'Loss', loss, 'Time', round(t, 3))
    print('*' * 100)
    if (e+1) % 5 == 0:
        print('-' * 100)
        seed = char_to_index(get_seed(sequence_length), char_idx)
        seq = model.generate_sequence(session, seed, 1000, sequence_length)
        print(index_to_chars(seq, char_idx))

        saver.save(session, '../checkpoints/shakespeare/1', global_step=e+1)

session.close()
