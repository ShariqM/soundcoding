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
    #return tf.truncated_normal([n_filter_width, 1, n_filters], 0.0,
            #tf.sqrt(2.0)/tf.sqrt(tf.cast(n_filter_width + n_filters, tf.float32)))
    return tf.truncated_normal([n_filter_width, n_filters], 0.0,
            tf.sqrt(2.0)/tf.sqrt(tf.cast(n_filter_width + n_filters, tf.float32)))


class Network(object):
    def __init__(self, n_filter_width, n_filters):
        self.F = tf.Variable(init_weights(n_filter_width, n_filters), name="F")
        return

    def step(self, x_input):
        #v_m = tf.conv1d(x_input, self.F, 1, 'SAME') # Add leak?
        # Add Noise

        v_rest  = tf.constant([0] * n_filters, name='v_rest')
        a_spike = tf.constant([1] * n_filters, name='a_spike')

        # Trainable?
        v_m = tf.Variable(tf.zeros(n_filters), trainable=True, name="v_m")
        a_m = tf.Variable(tf.zeros(n_filters), trainable=True, name="a_m")

        v_conv = tf.einsum("i,ij->j", x_input, self.F)
        #v_conv = tf.reduce_sum(tf.mul(x_input, F)) # Dot product
        #v_conv = tf.conv1d(x_input, self.F, 1, 'SAME')
        tf.scalar_summary(['v_m'] * n_filters, v_m)
        tf.scalar_summary(['a_m'] * n_filters, a_m)

        def spike_rest_op():
            return tf.tuple((v_m.assign(v_rest),
                             a_m.assign(a_spike)))

        def responding_op():
            return tf.tuple((v_m.assign(v_conv),
                            a_m))

        _step = tf.case((
                         #(tf.reshape(tf.greater(v_m, 0.5), [n_filters]), spike_rest_op),
                         (tf.greater(v_m, 0.5), spike_rest_op),
                        ),
                         responding_op)

        return _step



#batch = np.random.randn(batch_size, n_input, 1)
data = np.random.randn(n_input)

with tf.Graph().as_default(), tf.Session() as sess:
    network = Network(n_filter_width, n_filters)
    dataPH = tf.placeholder(tf.float32, shape=[n_filter_width], name="data")

    step_op = network.step(dataPH)
    summary_op = tf.merge_all_summaries()
    init_op = tf.initialize_all_variables()

    summary_writer = tf.train.SummaryWriter('AEresults', sess.graph)

    sess.run(init_op)
    for t in range(n_input - n_filter_width):
        feed_dict = {dataPH: data[t:t+n_filter_width]}
        sess.run(step_op, feed_dict=feed_dict)

        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, t)
        summary_writer.flush()

    x = np.random.randn(n_input)

    net = Network(n_filter_width, n_filters)
