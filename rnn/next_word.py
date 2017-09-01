import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


def get_inputs(scope='inputs'):
    """Create tf.placeholders for inputs, targets and learning rate"""
    with tf.name_scope(scope):
        inputs = tf.placeholder(tf.float32, [None, None], name='inputs')
        targets = tf.placeholder(tf.float32, [None, None], name='targets')


def get_cell(rnn_size, batch_size, layer_size):
    """Get a RNN cell with given input size and layers

    Args:
        rnn_size(int): The number of hidden units in the rnn cell.
        layer_size(int): The number of layers in the rnn.
    """
    # Set up the cell.
    cell = rnn.BasicLSTMCell(rnn_size)
    if layer_size > 1:
        cell = rnn.MultiRNNCell([cell]*layer_size)

    # Setup initial state.
    init_state = cell.zero_state(batch_size, tf.float32)
    init_state = tf.identity(init_state, 'initial_state')

    return cell, init_state


def build_rnn(cell, inputs):
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs)
    final_state = tf.identity(final_state, 'final_state')
    pass


def build_rnn(x, input_size, batch_size, layer_size):
    cell, init_state = get_cell(input_size, batch_size, layer_size)
    y, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=init_state)
    pass
    rnn.static_rnn()


def train():
    pass
