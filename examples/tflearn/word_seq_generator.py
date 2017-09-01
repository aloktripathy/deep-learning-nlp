from utils import SentenceReader
from gensim.models import Word2Vec
import numpy as np

from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import callbacks


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


def get_training_sequences(directory, word_index, max_seq_length=15, skip=3, unknown_word=-1):
    sequence = []
    reader = SentenceReader(directory)
    for words in reader:
        sequence += [word_index.get(word, unknown_word) for word in words]

    x, y = [], []
    for i in range(0, len(sequence)-max_seq_length, skip):
        x.append(sequence[i:i+max_seq_length])
        y.append(sequence[i+1:i+max_seq_length+1])
    return x, y


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
    print(word_sequence_matrix.shape)
    words = []
    for word_vector in word_sequence_matrix:
        word_str = w2v_model.wv.similar_by_vector(word_vector, topn=1)[0][0]
        words.append(word_str)
    print('-'*100)
    print(' '.join(words))
    print('-'*100)


WORD_2_VEC_MODEL = 'data/game_of_thrones/w2v.model'
DATA_DIRECTORY = 'data/game_of_thrones/small'
SEQ_LENGTH = 15
RNN_SIZE = 512
STATEFUL = True
BATCH_SIZE = 64
EPOCHS = 50

w2v_model = load_word_2_vec_model(WORD_2_VEC_MODEL)
v_size = w2v_model.vector_size
word_idx = get_word_index(w2v_model)

x, y = get_training_sequences(DATA_DIRECTORY, word_idx, max_seq_length=SEQ_LENGTH, skip=3)
x = sequences_to_vectors(x, SEQ_LENGTH, w2v_model)
y = sequences_to_vectors(y, SEQ_LENGTH, w2v_model)
# In a stateful network, you should only pass inputs with a number of samples that can be
# divided by the batch size.
truncated_input_size = x.shape[0] // BATCH_SIZE * BATCH_SIZE
x, y = x[:truncated_input_size, :, :], y[:truncated_input_size, :, :]


def get_random_sequence():
    idx = np.random.randint(len(y))
    return y[idx]


def generate_sequence(model, seq_length=100):
    sequence = np.zeros([BATCH_SIZE, seq_length, v_size], dtype=np.float32)
    sequence[0, :SEQ_LENGTH, :] = get_random_sequence()

    for idx in range(0, seq_length-SEQ_LENGTH+1):
        input_seq = sequence[:, idx:idx+SEQ_LENGTH, :]
        pred = model.predict(input_seq, batch_size=BATCH_SIZE)
        sequence[0, idx+SEQ_LENGTH-1, :] = pred[0, -1, :]
    vector_sequence_to_words(sequence[0])


model = models.Sequential()
model.add(
    layers.LSTM(RNN_SIZE, return_sequences=True, batch_input_shape=[BATCH_SIZE, SEQ_LENGTH, v_size],
                stateful=STATEFUL)
)
model.add(
    layers.LSTM(RNN_SIZE, return_sequences=True, input_shape=[SEQ_LENGTH, v_size], stateful=STATEFUL)
)
model.add(
    layers.LSTM(RNN_SIZE, return_sequences=True, input_shape=[SEQ_LENGTH, v_size], stateful=STATEFUL)
)
model.add(
    layers.LSTM(v_size, return_sequences=True, input_shape=[SEQ_LENGTH, v_size], stateful=STATEFUL)
)
model.compile('rmsprop', 'cosine', metrics=['accuracy'])

filepath = "checkpoints/got/1"
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

for e in range(EPOCHS):
    model.fit(x, y, batch_size=64, epochs=1)
    generate_sequence(model, 100)
    if (e + 1) % 5 == 0:
        model.save('checkpoints/got/{}.h5'.format(e))
