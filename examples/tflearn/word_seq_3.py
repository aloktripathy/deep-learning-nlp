from utils import SentenceReader
from gensim.models import Word2Vec
import numpy as np
import tflearn


def generate_word_2_vec(directory, filename, min_count=5, vector_size=50):
    sentences = SentenceReader(directory)
    model = Word2Vec(sentences, min_count=min_count, size=vector_size)
    model.save(filename)
    # generate_word_2_vec(
    #     '../../data/got/small',
    #     '../../data/got/small/w2v.model',
    # )


def load_word_2_vec_model(filename):
    return Word2Vec.load(filename)


def get_word_index(w2v_model):
    return {word: idx for idx, word in enumerate(w2v_model.wv.index2word)}


def get_training_sequences(directory, word_index, max_seq_length=15, unknown_word=-1):
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


WORD_2_VEC_MODEL = 'data/got/w2v.model'
DATA_DIRECTORY = 'data/got/small'
SEQ_LENGTH = 15
RNN_SIZE = 512
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


g = tflearn.input_data(shape=[None, SEQ_LENGTH])
g = tflearn.embedding(g, input_dim=VOCAB_SIZE, output_dim=N_VECTORS, name='embedding_layer')
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512)
g = tflearn.dropout(g, 0.5)
g = tflearn.fully_connected(g, VOCAB_SIZE, activation='softmax')
g = tflearn.regression(g, optimizer='rmsprop', loss='categorical_crossentropy', learning_rate=0.001)

m = tflearn.SequenceGenerator(g, dictionary=word_idx,
                              seq_maxlen=SEQ_LENGTH,
                              clip_gradients=5.0,
                              max_checkpoints=3,
                              checkpoint_path='model_data/got/tfl/')

embedding_weights = tflearn.get_layer_variables_by_name('embedding_layer')[0]
m.set_weights(embedding_weights, w2v_model.wv.syn0)


def train(epochs):
    for i in range(epochs):
        seed = get_random_sequence()
        m.fit(x, y, validation_set=0.1, batch_size=128,
              n_epoch=1, run_id='got')
        print("-- TESTING...")
        print("-- Test with temperature of 1.2 --")
        print(m.generate(30, temperature=1.2, seq_seed=seed))
        print("-- Test with temperature of 1.0 --")
        print(m.generate(30, temperature=1.0, seq_seed=seed))
        print("-- Test with temperature of 0.5 --")
        print(m.generate(30, temperature=0.5, seq_seed=seed))
