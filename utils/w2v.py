import os
import sys

from gensim.models import Word2Vec
import numpy as np

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

from utils.string import SentenceReader


def generate_word_2_vec(directory, filename, min_count=5, vector_size=50):
    sentences = SentenceReader(directory)
    model = Word2Vec(sentences, min_count=min_count, size=vector_size)
    model.save(filename)
    # generate_word_2_vec(
    #     '../../data/got/small',
    #     '../../data/got/small/w2v.model',
    # )
    return model


def load_word_2_vec_model(filename):
    return Word2Vec.load(filename)


def get_word_index(w2v_model):
    return {word: idx for idx, word in enumerate(w2v_model.wv.index2word)}


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
