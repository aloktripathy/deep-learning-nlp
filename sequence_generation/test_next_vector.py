from sequence_generation.next_vector import *

vector_size = 50
data_dir = os.path.join(ROOT_PATH, 'data/{}'.format(DATA_SET))
model_dir = os.path.join(ROOT_PATH, 'model_data/{}'.format(DATA_SET))
w2v_model_file = os.path.join(model_dir, '{}.w2v'.format(vector_size))

w2v_model = load_word_2_vec_model(w2v_model_file)

n_vectors = w2v_model.vector_size
vocab_size = len(w2v_model.wv.vocab)
word_idx = get_word_index(w2v_model)
x_train, y_train, x_val, y_val = get_data_set(data_dir, BATCH_SIZE, word_idx, SEQ_LENGTH)


GEN_SEQ_LEN = 100
TEMPERATURE = 100

model, e = load_keras_model('../model_data/got/small/')
sequence = np.zeros([BATCH_SIZE, GEN_SEQ_LEN], dtype=np.int32)
sequence[0, :SEQ_LENGTH] = get_random_sequence(x_train)

input_seq = sequence[:, 0:SEQ_LENGTH]

pred = model.predict(input_seq, batch_size=BATCH_SIZE)

print('Actual sentence -')
print(indexes_to_words(sequence[0], w2v_model))

print('What the NN thinks -')
print(' '.join([w2v_model.wv.index2word[i.argmax()] for i in pred[0]]))

n = 0
for i in pred[0][-1]:
    if i == 0:
        n += 1

print('# of words with zero probability to become next word: {}'.format(n))
generate_sequence(x_train, w2v_model, model, GEN_SEQ_LEN, TEMPERATURE)
