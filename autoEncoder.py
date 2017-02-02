import tensorflow as tf
import pdb
import numpy as np















# Network Parameters
N = 1000 # Dim of I
n_input = 1000
n_filters = 48
n_filter_width = 128
n_sample_rate = 16000
batch_size = 128

def init_weights(n_filter_width, n_filters):
    return tf.truncated_normal([n_filter_width, 1, n_filters], 0.0,
            tf.sqrt(2.0)/tf.sqrt(tf.cast(n_filter_width + n_filters, tf.float32)))


class Network(object):
    def __init__(self, n_filter_width, n_filters):
        self.F = tf.Variable(init_weights(n_filter_width, n_filters), name="F")
        pass

    def step(self, x_input):
        out = tf.conv1d(x_input, self.F, 1, 'SAME')

batch = np.random.randn(batch_size, n_input, 1)

net = Network(n_filter_width, n_filters)
