import tensorflow as tf
import pdb

def init_encode_weights(n_filter_width, n_filters):
    return tf.truncated_normal((n_filter_width, 1, n_filters), 0.0,
            1/tf.sqrt(tf.cast(n_filter_width, tf.float32)))

def init_decode_weights(n_filter_width, n_filters):
    return tf.truncated_normal((n_filter_width, n_filters, 1), 0.0,
            1/tf.sqrt(tf.cast(n_filter_width, tf.float32)))

class AutoEncoder(object):
    def __init__(self, model):
        self.model = model
        n_filters, n_filter_width = model.n_filters, model.n_filter_width
        self.A = tf.Variable(init_encode_weights(n_filter_width, n_filters), name="analysis_filters")
        self.S = tf.Variable(init_decode_weights(n_filter_width, n_filters), name="synthesis_filters")

    def get_filters_ph(self):
        return self.A, self.S

    def encode(self, x_input, noise):
        u = tf.nn.conv1d(x_input, self.A, 1, padding='SAME')
        u_var = tf.square(u)
#
        ##n = 5 # Number of points in the past to consider
        ##c = 3 # Controls capacity through the ratio of variance of noise to variance of response
        #variances = tf.nn.conv1d(u2, tf.Variable(np.ones(n) * 1./(n+c)))
        #noise = tf.random_normal([model.n_batch_size, model.n_steps, 1], mean=0.0, stddev=tf.sqrt(variances))


        r = u + noise
        #return u_var, r
        a = tf.nn.relu(r)
        return u_var, a

    def channel(self, u, noise):
        return

    def decode(self, a):
        return tf.nn.conv1d(a, self.S, 1, padding='SAME')
