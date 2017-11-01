"""
In this model, we try to predict the next word vector given a sequence of word vectors.
"""

import os
import sys
import numpy as np

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import CuDNNGRU
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Embedding
from keras.layers import TimeDistributed
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

from sequence_generation.gen_utils import indexes_to_words
from sequence_generation.gen_utils import temperature_sample
from utils.w2v import get_word_index
from utils.w2v import load_word_2_vec_model
from utils.w2v import generate_word_2_vec
from utils.training import get_streamlined_training_sequences
from utils.training import SaveModel
from utils.training import RelaxSystem
from utils.training import ResetStates
from utils.training import load_keras_model

import pandas as pd
pd.set_option("display.width", 500)
pd.set_option("display.max_rows", 500)


def get_data_set(data_directory, batch_size, word_idx, seq_length):
    x, y = get_streamlined_training_sequences(data_directory, word_idx, batch_size, seq_length)
    data = train_test_split(x, y, test_size=0.1)
    return tuple(tensor.reshape(-1, seq_length) for tensor in data)


def get_random_sequence(a):
    idx = np.random.randint(len(a))
    return a[idx]


def generate_sequence(x, w2v, model, seq_length=100, temperature=0.5, debug=False):
    model.reset_states()
    sequence = np.zeros([BATCH_SIZE, seq_length], dtype=np.int32)
    sequence[0, :SEQ_LENGTH] = get_random_sequence(x)
    debugging_list = list()
    # debugging_list.append([w2v.wv.index2word[i] for i in sequence[0, :SEQ_LENGTH]])
    for idx in range(0, seq_length-SEQ_LENGTH):
        input_seq = sequence[:, idx:idx+SEQ_LENGTH]
        pred = model.predict(input_seq, batch_size=BATCH_SIZE)
        debugging_list.append([w2v.wv.index2word[i] for i in sequence[0, idx:idx+SEQ_LENGTH]])
        # debugging_list.append([w2v.wv.index2word[i.argmax()] for i in pred[0]])
        pred_idx = pred[0][-1].argmax()# temperature_sample(pred[0][-1], temperature)
        sequence[0, idx+SEQ_LENGTH] = pred_idx
    if debug:
        print('INSIDE NN ->')
        df = pd.DataFrame(debugging_list)
        print(df)
    word_seq = indexes_to_words(sequence[0], w2v)
    delim = '\n'+'-'*100+'\n'
    print(delim, word_seq, delim)

    word_seq += '\n' + '-'*100
    with open('generated.txt', 'a+') as fp:
        fp.write(word_seq)


def build_model(w2v_model, rnn_size, vocab_size, n_vectors, seq_length, batch_size,
                is_stateful, is_embedding_trainable):
    weights = w2v_model.wv.syn0
    model = Sequential()
    model.add(
        Embedding(input_dim=vocab_size, output_dim=n_vectors, input_length=seq_length, weights=[weights],
                  batch_input_shape=[batch_size, seq_length], trainable=is_embedding_trainable),
    )
    model.add(CuDNNGRU(rnn_size, return_sequences=True, stateful=is_stateful))
    model.add(CuDNNGRU(rnn_size, return_sequences=True, stateful=is_stateful))
    model.add(CuDNNGRU(rnn_size, return_sequences=True, stateful=is_stateful))
    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Activation('softmax'))
    model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train(model, w2v_model, epochs, batch_size, seq_length, x_train, y_train, x_val, y_val,
          model_dir, model_name, save_every=5):
    pre_trained_model, initial_epoch = load_keras_model(model_dir)
    if pre_trained_model:
        print('loading pre-trained model from epoch {}...'.format(initial_epoch))
        model = pre_trained_model

    tensorboard = TensorBoard(
        log_dir='/var/log/keras/seq_generation/{}'.format(model_name),
        histogram_freq=0,
        batch_size=64,
        write_graph=pre_trained_model is None,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None
    )
    lr_reducer = ReduceLROnPlateau(mode='min', patience=3, verbose=1)
    model_saver = SaveModel(save_every, model, model_dir)
    sequence_generator = GenerateSequence(x_train, w2v_model, model, 200)
    system_relaxer = RelaxSystem(60*20, 60*3)
    reset_states = ResetStates(model)
    callbacks = [reset_states, tensorboard, lr_reducer, model_saver, sequence_generator,
                 system_relaxer]
    # Load the model for resuming training if a pre-trained model already exists.

    model.fit(
        x_train,
        y_train.reshape(-1, seq_length, 1),
        batch_size=batch_size,
        epochs=epochs,
        initial_epoch=initial_epoch or 0,
        validation_data=(x_val, y_val.reshape(-1, seq_length, 1)),
        callbacks=callbacks,
        shuffle=False
    )


SEQ_LENGTH = 20
RNN_SIZE = 256
BATCH_SIZE = 64
EPOCHS = 150
N_VECTORS = None
VOCAB_SIZE = None
IS_EMBEDDING_TRAINABLE = True
IS_STATEFUL = True
DATA_SET = 'got/main'
MODEL_NAME = 'got-main-w2v-3l-{}-{}-{}-{}'.format(
    RNN_SIZE,
    SEQ_LENGTH,
    'trainable' if IS_EMBEDDING_TRAINABLE else 'non-trainable',
    'stateful' if IS_STATEFUL else 'stateless'
)


class GenerateSequence(Callback):
    def __init__(self, x, w2v_model, model, seq_length):
        super(GenerateSequence, self).__init__()
        self._x = x
        self._model = model
        self._w2v_model = w2v_model
        self._seq_length = seq_length

    def on_epoch_end(self, epoch, logs=None):
        generate_sequence(self._x, self._w2v_model, self._model, self._seq_length)


def run():
    vector_size = 50
    data_dir = os.path.join(ROOT_PATH, 'data/{}'.format(DATA_SET))
    model_dir = os.path.join(ROOT_PATH, 'model_data/{}'.format(DATA_SET))
    w2v_model_file = os.path.join(model_dir, '{}.w2v'.format(vector_size))
    try:
        w2v_model = load_word_2_vec_model(w2v_model_file)
    except FileNotFoundError:
        print('Word2Vec model not found. Creating it now.')
        w2v_model = generate_word_2_vec(data_dir, w2v_model_file, min_count=5, vector_size=vector_size)

    n_vectors = w2v_model.vector_size
    vocab_size = len(w2v_model.wv.vocab)
    word_idx = get_word_index(w2v_model)
    x_train, x_val, y_train, y_val = get_data_set(data_dir, BATCH_SIZE, word_idx, SEQ_LENGTH)
    # x_train, y_train = x_train[:1024, :], y_train[:1024]
    model = build_model(w2v_model, RNN_SIZE, vocab_size, n_vectors, SEQ_LENGTH, BATCH_SIZE,
                        IS_STATEFUL, IS_EMBEDDING_TRAINABLE)

    train(model, w2v_model, EPOCHS, BATCH_SIZE, SEQ_LENGTH, x_train, y_train, x_val, y_val, model_dir, MODEL_NAME, 5)


if __name__ == "__main__":
    run()
