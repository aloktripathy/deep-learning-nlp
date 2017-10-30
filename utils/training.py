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
