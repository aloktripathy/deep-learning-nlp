import tflearn
from tflearn.layers.core import input_data
from tflearn.layers.core import dropout
from tflearn.layers.core import fully_connected
from tflearn.layers.conv import conv_2d
from tflearn.layers.conv import max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.models import DNN


tflearn.init_graph(seed=8888, num_cores=16, gpu_memory_fraction=0.5)

# This is a deep conv net.
# The activation size for each layer is computed as follows
# Where A is the activation size, W is the input volume size, P is the zero-adding size and S is
# stride size.
# A = (W − F + 2P) / S + 1

# Fetch the inputs.
x, y, x_val, y_val = 1, 2, 3, 4

network = input_data([None, 3, 32, 32], name='input-data')

# Activation: [None, 16, 28, 28], weights: [16, 3, 5, 5]
network = conv_2d(network, nb_filter=16, filter_size=5, activation='relu')

# Activation: [None, 16, 13, 13]
network = max_pool_2d(network, kernel_size=3, strides=2)

# Activation: [None, 32, 10, 10], weights: [32, 16, 5, 5]
network = conv_2d(network, nb_filter=32, filter_size=4, activation='relu')
network = dropout(network, keep_prob=0.75)

# Activation: [None, 64, 8, 8], weights : [64, 32, 3, 3]
network = conv_2d(network, nb_filter=64, filter_size=3, activation='relu')
network = dropout(network, keep_prob=0.5)

# Activation: [None, 512], weights: [512, 64 * 8 * 8]
network = fully_connected(network, 512, activation='relu')
network = dropout(network, keep_prob=0.5)

# Activations: [None, 10], weights: [10, 512]
softmax = fully_connected(network, 10, activation='softmax')
network = dropout(network, keep_prob=0.5)

regression = regression(softmax, optimizer='rmsprop', loss='categorical_crossentropy')

model = DNN(regression, max_checkpoints=3, tensorboard_verbose=1)

model.load('cifar10_cnn')

model.fit(x, y, n_epoch=10, validation_set=0.1, shuffle=True, show_metric=True, batch_size=64,
          run_id='cifar_10_dnn')

model.save('model-fine-tuning')
