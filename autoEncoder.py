import tensorflow as tf
import pdb

def init_encode_weights(n_filter_width, n_filters):
    return tf.truncated_normal((n_filter_width, 1, n_filters), 0.0,
            1/tf.sqrt(tf.cast(n_filter_width, tf.float32)))

def init_decode_weights(n_filter_width, n_filters):
    return tf.truncated_normal((n_filter_width, n_filters, 1), 0.0,
            1/tf.sqrt(tf.cast(n_filter_width, tf.float32)))

class AutoEncoder(object):
    def __init__(self, n_filter_width, n_filters):
        self.A = tf.Variable(init_encode_weights(n_filter_width, n_filters), name="analysis_filters")
        self.S = tf.Variable(init_decode_weights(n_filter_width, n_filters), name="synthesis_filters")

    def get_filters_ph(self):
        return self.A, self.S

    def encode(self, x_input, noise):
        u = tf.nn.conv1d(x_input, self.A, 1, padding='SAME')

        r = u + noise
        a = tf.nn.relu(r)
        return a

    def decode(self, a):
        return tf.nn.conv1d(a, self.S, 1, padding='SAME')
