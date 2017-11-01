import glob
import os
import sys
import time

import numpy as np
from keras.callbacks import Callback
from keras.models import load_model

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

from utils.string import SentenceReader
from utils.string import natural_sort


def get_training_sequences(directory, word_index, max_seq_length=15):
    sequence = []
    reader = SentenceReader(directory)
    for words in reader:
        for word in words:
            index = word_index.get(word)
            # Remove unknown words.
            if index is None:
                continue
            sequence.append(index)

    x, y = [], []
    for i in range(0, len(sequence)-max_seq_length):
        x.append(sequence[i:i+max_seq_length])
        y.append(sequence[i+1:i+max_seq_length+1])
    return np.array(x, dtype=np.uint16), np.array(y, dtype=np.uint16)


def get_streamlined_training_sequences(directory, word_index, batch_size, seq_length):
    sequence = []
    reader = SentenceReader(directory)
    for words in reader:
        for word in words:
            index = word_index.get(word)
            # Remove unknown words.
            if index is None:
                continue
            sequence.append(index)

    # Perform batching.
    return make_streamlined_batches(np.array(sequence, dtype=np.int16), batch_size, seq_length)


def make_streamlined_batches(sequence, batch_size, seq_length):
    n = sequence.shape[0] // batch_size * batch_size
    sequence = sequence[:n]
    batched_seq = sequence.reshape([64, -1])
    n_batches = batched_seq.shape[1] - seq_length + 1
    x = np.zeros([n_batches - 1, batch_size, seq_length], dtype=sequence.dtype)
    y = np.zeros([n_batches - 1, batch_size, seq_length], dtype=sequence.dtype)
    for i in range(n_batches - 1):
        x[i, :, :] = batched_seq[:, i:i + seq_length]
        y[i, :, :] = batched_seq[:, i + 1:i + seq_length + 1]
    return x, y


class SaveModel(Callback):
    """This is a Keras callback that saves a model after n epochs."""
    def __init__(self, save_every, model, model_dir):
        super(SaveModel, self).__init__()
        self._save_every = save_every
        self._model = model
        self._model_dir = model_dir

    def on_epoch_end(self, epoch, logs=None):
        epoch = epoch + 1
        if epoch % self._save_every == 0:
            path = os.path.join(self._model_dir, '{}.h5'.format(epoch))
            self._model.save(path)


class RelaxSystem(Callback):
    """Keras callback that halts training for few minutes periodically so that GPU can cool off."""
    def __init__(self, relax_every=60 * 60, relaxing_duration=60 * 5):
        """
        Args:
            relax_every(int): Seconds after which system should relax.
            relaxing_duration(int): Duration in seconds for which system will relax.
        """
        super(RelaxSystem, self).__init__()
        self._cool_off_every = relax_every
        self._cooling_duration = relaxing_duration
        self._last_relaxed_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if time.time() - self._last_relaxed_time > self._cool_off_every:
            print('System is relaxing for next {} seconds...'.format(self._cooling_duration))
            time.sleep(self._cooling_duration)
            self._last_relaxed_time = time.time()


class ResetStates(Callback):
    """This callback resets the LSTM states of a model before training for an epoch."""
    def __init__(self, model):
        super(ResetStates, self).__init__()
        self._model = model

    def on_epoch_end(self, epoch, logs=None):
        self._model.reset_states()


def load_keras_model(model_dir):
    """Given a model directory, get the most recently trained model to resume training from."""
    glob_regex = os.path.join(model_dir, '*.h5')
    files = glob.glob(glob_regex)
    files = natural_sort(files)
    if not files:
        return None, None

    most_recent_file = files[-1]
    model = load_model(most_recent_file)
    file_name = most_recent_file.split('/')[-1]
    epoch_count = int(file_name.split('.')[0])

    return model, epoch_count
