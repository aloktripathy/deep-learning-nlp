import os
import numpy as np

from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers


class RNNLearn(metaclass=ABCMeta):
    """This is a base class implementation of RNN cells.

    Notes:
        - Basics on tensorflow - http://web.stanford.edu/class/cs20si/lectures/slides_02.pdf
    """
    def __init__(self, rnn_size, layer_size, batch_size):
        """
        Args:
            layer_size(int): The number of layers stacked in the RNN architecture.
            batch_size(int): The number of sequences in each batch. Useful for getting initial state.
        """
        self._RNN_SIZE = rnn_size
        self._LAYER_SIZE = layer_size
        self._BATCH_SIZE = batch_size

        # Create placeholders for input and output.
        self._x, self._y, self._batch_size = self._get_inputs()

        # Apply internal transformations that you needs to be applied before ingested to the net.
        self._x_runnable, self._y_runnable = self._prepare_inputs()

        # Build the rnn.
        self._rnn_output, self._init_state, self._final_state, self._rnn_cell = self._get_rnn()
        self._y_pred = self._build_nn()

        # Attributes for training setup.
        self._is_set_for_training = False
        self._learning_rate = None
        self._optimization_function = None
        self._train_step = None
        self._loss = None
        self._accuracy = None

    def train(self, session, x_batch, y_batch, state):
        assert self._is_set_for_training, 'Model not set for training. Please call ' \
                                          'set_learning_rate() method to do training setup.'
        feed_dict = self._get_feed_dict(session, state, x_batch, y_batch)
        feed_dict[self._batch_size] = self._BATCH_SIZE

        _, loss, final_state = session.run(
            [self._train_step, self._loss, self._final_state],
            feed_dict=feed_dict
        )
        return final_state, loss

    def track_progress(self):
        pass

    def predict(self, session, state, x):
        feed_dict = self._get_feed_dict(session, state, x)
        y_pred, out_state = session.run([self._y_pred, self._final_state], feed_dict=feed_dict)
        return y_pred, out_state

    def get_accuracy(self, session, state, x, y):
        feed_dict = self._get_feed_dict(session, state, x, y)
        return session.run(self._accuracy, feed_dict)

    @abstractmethod
    def setup_training(self, learning_rate=0.002):
        self._learning_rate = learning_rate
        self._optimization_function = tf.train.AdamOptimizer(self._learning_rate)
        self._is_set_for_training = True

    def _get_feed_dict(self, session, state, x, y=None):
        feed_dict = {self._x: x}
        if y is not None:
            feed_dict[self._y] = y
        if state is None:
            state = session.run(self._init_state)

        # Feed the input states.
        for idx, state_op in enumerate(self._init_state):
            feed_dict[state_op] = state[idx]
        return feed_dict

    @abstractmethod
    def _build_nn(self):
        raise NotImplementedError()

    @abstractmethod
    def _prepare_inputs(self):
        raise NotImplementedError()

    def _get_rnn(self):
        """Build the RNN network."""
        if self._LAYER_SIZE != 1:
            cell = rnn.MultiRNNCell([self._create_cell() for _ in range(self._LAYER_SIZE)])
        else:
            cell = self._create_cell()

        init_state = cell.zero_state(self._BATCH_SIZE, tf.float32)

        # output      : [batch_size, seq_length, rnn_size]
        # final_state : [batch_size, layer_size * rnn_size]
        output, final_state = tf.nn.dynamic_rnn(cell, self._x_runnable, initial_state=init_state)

        # Give the variables names so that we can fetch them from checkpoints.
        final_state = tf.identity(final_state, 'final-state')
        return output, init_state, final_state, cell

    def _create_cell(self):
        """Return an RNN cell with rnn_size number of units."""
        return rnn.GRUCell(self._RNN_SIZE)

    def _get_inputs(self):
        """Get the tf placeholders for input variables."""
        # This is going to hold a batch of input sequences.
        # for example -
        # [[1, 2, 3], [7, 8, 9], [13, 14, 15], [19, 20, 21]]
        x = tf.placeholder(tf.uint8, [None, None], name='x')

        # Hold a batch of targets with same sequence length as x.
        y = tf.placeholder(tf.uint8, [None, None], name='y')

        #
        batch_size = tf.placeholder(tf.int32, name='batch-size')
        return x, y, batch_size

    def _restore_model(self, session, path, graph_file_name):
        """Restore a trained model.

        Args:
            session(tensorflow.Session): A tensorflow session.
            path(str): The file path where the model is stored. e.g. - '../../shakespeare/'
            graph_file_name(str): The graph file name inside `path`. e.g. - '1-40.meta'

        Returns:
            saver.
        """
        graph_file = os.path.join(path, graph_file_name)
        if not (os.path.exists(path) and os.path.isfile(graph_file)):
            print('Could not restore model.')
            return
        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(graph_file)
        saver.restore(session, tf.train.latest_checkpoint(path))
        return saver


class NextChar(RNNLearn):
    def __init__(self, alphabet_size, rnn_size, layer_size, batch_size):
        self.alphabet_size = alphabet_size
        self._y_logits = None
        super().__init__(rnn_size, layer_size, batch_size)

    def _prepare_inputs(self):
        # x_1_hot, y_1_hot: [batch_size, seq_length, alphabet_size]
        x_1_hot = tf.one_hot(self._x, self.alphabet_size, 1., 0., dtype=tf.float32, name='x-one-hot')
        y_1_hot = tf.one_hot(self._y, self.alphabet_size, 1., 0., dtype=tf.float32, name='y-one-hot')
        return x_1_hot, y_1_hot

    def _build_nn(self):
        with tf.name_scope('output'):
            # Flatten output to make it a tensor of shape [batch_size * seq_length, rnn_size]
            # This will help us apply the softmax function.
            output = tf.reshape(self._rnn_output, [-1, self._RNN_SIZE])

        with tf.name_scope('fully-connected'):
            # This is a fully connected layer without activation function. i.e. activation
            # function is the the sum of multiplication of inputs and weights.
            # [batch_size * seq_length, alphabet_size]
            y_logits = layers.linear(output, self.alphabet_size)

        with tf.name_scope('prediction'):
            y_1_hot_pred = tf.nn.softmax(y_logits)  # [batch_size * seq_length, alphabet_size]
            y_pred = tf.argmax(y_1_hot_pred, 1)  # [batch_size * seq_length, alphabet_size]
            y_pred = tf.reshape(y_pred, [self._batch_size, -1], name='y_pred')  # [batch_size, seq_length]

        self._y_logits = y_logits
        return y_pred

    def setup_training(self, learning_rate=0.002):
        super().setup_training(learning_rate)
        with tf.name_scope('train-step'):
            # Apply softmax.
            # [batch_size * seq_length, alphabet_size]
            y_1_hot_flat = tf.reshape(self._y_runnable, [-1, self.alphabet_size])
            # [batch_size * seq_length]
            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=self._y_logits, labels=y_1_hot_flat
            )
            # [batch_size, seq_length]
            loss = tf.reshape(loss, [self._batch_size, -1])
            train_step = self._optimization_function.minimize(loss, name='train-step')
            tf.add_to_collection("optimizer", train_step)

        # Stats for display.
        with tf.name_scope('accuracy-and-loss'):
            seq_loss = tf.reduce_mean(loss, 1)  # [batch_size]
            batch_loss = tf.reduce_mean(seq_loss, name='batch-loss')
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self._y, tf.cast(self._y_pred, tf.uint8)), tf.float32),
                name='accuracy'
            )

        self._train_step, self._loss, self._accuracy = train_step, batch_loss, accuracy

    def generate_sequence(self, session, seed, length, seq_length):
        """Given a seed sequence, complete the sequence for n characters.

        Args:
            seed(iterable): A list / tuple of character numbers.
            length(int): Total size of the generated sequence.
            seq_length(int): Sequence window.
        """
        sequence = list(seed)
        if self._is_set_for_training:
            batch_size = self._BATCH_SIZE
        else:
            batch_size = 1
        state = tuple(
            np.zeros([batch_size, self._RNN_SIZE], dtype=np.float32) for _ in range(self._LAYER_SIZE)
        )
        # zero_state = self._rnn_cell.zero_state(batch_size, dtype=tf.float32)
        for i in range(length):
            x_seq = np.reshape(sequence[-seq_length:], [1, -1])
            x = np.zeros([batch_size, seq_length])
            x[0, :x_seq.shape[1]] = x_seq
            feed_dict = {self._x: x, self._batch_size: batch_size}
            for idx, op in enumerate(self._init_state):
                feed_dict[op] = state[idx]

            pred = session.run(self._y_pred, feed_dict)
            sequence.append(pred[0][-1])
        return sequence

    def restore_model(self, session, path, graph_file_name):
        saver = self._restore_model(session, path, graph_file_name)

        if not saver:
            return

        # Load all tensors here.
        self._x = session.graph.get_tensor_by_name('x:0')
        self._y = session.graph.get_tensor_by_name('y:0')
        self._batch_size = session.graph.get_tensor_by_name('batch-size:0')

        # Apply internal transformations that you needs to be applied before ingested to the net.
        self._x_runnable = session.graph.get_tensor_by_name('x-one-hot:0')
        self._y_runnable = session.graph.get_tensor_by_name('y-one-hot:0')

        # Build the rnn.
        self._final_state = session.graph.get_tensor_by_name('final-state:0')
        self._y_pred = session.graph.get_tensor_by_name('prediction/y_pred:0')

        # Attributes for training setup.
        self._train_step = session.graph.get_collection('optimizer', 'train-step')[0]
        self._loss = session.graph.get_tensor_by_name('accuracy-and-loss/batch-loss:0')
        self._accuracy = session.graph.get_tensor_by_name('accuracy-and-loss/accuracy:0')

        return saver
