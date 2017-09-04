from utils import SentenceReader
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


def generate_word_2_vec(directory, filename, min_count=5, vector_size=50):
    sentences = SentenceReader(directory)
    model = Word2Vec(sentences, min_count=min_count, size=vector_size)
    model.save(filename)
    # generate_word_2_vec(
    #     '../../data/game_of_thrones/small',
    #     '../../data/game_of_thrones/small/w2v.model',
    # )


def load_word_2_vec_model(filename):
    return Word2Vec.load(filename)


def get_word_index(w2v_model):
    return {word: idx for idx, word in enumerate(w2v_model.wv.index2word)}


def get_training_sequences(directory, word_index, max_seq_length=15, unknown_word=0):
    sequence = []
    reader = SentenceReader(directory)
    for words in reader:
        sequence += [word_index.get(word, unknown_word) for word in words]

    x, y = [], []
    for i in range(0, len(sequence)-max_seq_length):
        x.append(sequence[i:i+max_seq_length])
        y.append(sequence[i+1:i+max_seq_length+1])
    return np.array(x, dtype=np.uint16), np.array(y, dtype=np.uint16)


def sequences_to_vectors(sequences, seq_length, w2v_model):
    """Convert a batch of sequences to word vectors."""
    batch_size = len(sequences)
    vec_matrix = np.zeros([batch_size, seq_length, w2v_model.vector_size], dtype=np.float32)
    for i in range(batch_size):
        for j in range(seq_length):
            word_index = sequences[i][j]
            if word_index == -1:
                continue
            vector = w2v_model.wv[w2v_model.wv.index2word[word_index]]
            vec_matrix[i, j, :] = vector.reshape(1, w2v_model.vector_size)
    return vec_matrix


def vector_sequence_to_words(word_sequence_matrix):
    words = []
    for word_vector in word_sequence_matrix:
        word_str = w2v_model.wv.similar_by_vector(word_vector, topn=1)[0][0]
        words.append(word_str)
    print('-'*100)
    print(' '.join(words))
    print('-'*100)


def indexes_to_words(word_index_sequence):
    words = []
    for idx in word_index_sequence:
        words.append(w2v_model.wv.index2word[idx])
    print('-' * 100)
    print(' '.join(words))
    print('-' * 100)


WORD_2_VEC_MODEL = 'data/game_of_thrones/w2v.model'
DATA_DIRECTORY = 'data/game_of_thrones/small'
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


def generate_sequence(model, seq_length=100):
    sequence = np.zeros([BATCH_SIZE, seq_length], dtype=np.int32)
    sequence[0, :SEQ_LENGTH] = get_random_sequence()

    for idx in range(0, seq_length-SEQ_LENGTH+1):
        input_seq = sequence[:, idx:idx+SEQ_LENGTH]
        pred = model.predict(input_seq, batch_size=BATCH_SIZE)
        pred_idx = np.argmax(pred[0][-1])
        sequence[0, idx+SEQ_LENGTH-1] = pred_idx
    indexes_to_words(sequence[0])


def build_model():
    weights = w2v_model.wv.syn0
    model = Sequential()
    model.add(
        Embedding(input_dim=VOCAB_SIZE, output_dim=N_VECTORS,
                  weights=[weights], input_length=SEQ_LENGTH,
                  batch_input_shape=[BATCH_SIZE, SEQ_LENGTH], trainable=False),
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
                  verbose=2, callbacks=[tensorboard])
        generate_sequence(model, 200)
        if (e + 1) % 5 == 0:
            model.save('checkpoints/got/{}.h5'.format(e+1))


if __name__ == '__main__':
    model = build_model()
    train(model)
