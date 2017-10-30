import os
import sys
from gensim.models import Word2Vec
import numpy as np

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Embedding
from keras.layers import TimeDistributed
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard


ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

from utils import SentenceReader


def vector_sequence_to_words(word_sequence_matrix):
    words = []
    for word_vector in word_sequence_matrix:
        word_str = w2v_model.wv.similar_by_vector(word_vector, topn=1)[0][0]
        words.append(word_str)
    print('-'*100)
    print(' '.join(words))
    print('-'*100)





WORD_2_VEC_MODEL = '../data/got/w2v.model'
DATA_DIRECTORY = '../data/got/small'
SEQ_LENGTH = 25
RNN_SIZE = 256
STATEFUL = True
BATCH_SIZE = 64
EPOCHS = 50
N_VECTORS = None
VOCAB_SIZE = None


w2v_model = load_word_2_vec_model(WORD_2_VEC_MODEL)
N_VECTORS = w2v_model.vector_size
VOCAB_SIZE = len(w2v_model.wv.vocab)
word_idx = get_word_index(w2v_model)

x, y = get_training_sequences(DATA_DIRECTORY, word_idx, max_seq_length=SEQ_LENGTH)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=2112)
# del x
# del y
# In a stateful network, you should only pass inputs with a number of samples that can be
# divided by the batch size.
truncated_input_size_train = x_train.shape[0] // BATCH_SIZE * BATCH_SIZE
truncated_input_size_val = x_val.shape[0] // BATCH_SIZE * BATCH_SIZE
x_train, y_train = x_train[:truncated_input_size_train, :], y_train[:truncated_input_size_train, :]
x_val, y_val = x_val[:truncated_input_size_val, :], y_val[:truncated_input_size_val, :]


def get_random_sequence():
    idx = np.random.randint(len(x))
    return x[idx]







def build_model():
    # weights = w2v_model.wv.syn0
    model = Sequential()
    model.add(
        Embedding(input_dim=VOCAB_SIZE, output_dim=N_VECTORS, input_length=SEQ_LENGTH,
                  batch_input_shape=[BATCH_SIZE, SEQ_LENGTH], trainable=True),
    )
    model.add(LSTM(RNN_SIZE, return_sequences=True, stateful=STATEFUL))
    model.add(LSTM(RNN_SIZE, return_sequences=True, stateful=STATEFUL))
    model.add(LSTM(RNN_SIZE, return_sequences=True, stateful=STATEFUL))
    model.add(TimeDistributed(Dense(VOCAB_SIZE)))
    model.add(Activation('softmax'))
    model.compile('rmsprop', 'sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def to_one_hot(sequences):
    return to_categorical(sequences, VOCAB_SIZE).reshape(-1, SEQ_LENGTH, VOCAB_SIZE)


def train(model):
    tensorboard = TensorBoard(
        log_dir='/home/ubuntu/log/keras/got/small',
        histogram_freq=0,
        batch_size=64,
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None
    )

    for e in range(EPOCHS):
        model.fit(x_train, y_train.reshape(-1, SEQ_LENGTH, 1), batch_size=BATCH_SIZE,
                  epochs=1, validation_data=(x_val, y_val.reshape(-1, SEQ_LENGTH, 1)),
                  # callbacks=[tensorboard]
                  )
        generate_sequence(model, 200)
        if (e + 1) % 5 == 0:
            model.save('model_data/got/{}.h5'.format(e+1))


if __name__ == '__main__':
    model = build_model()
    train(model)
