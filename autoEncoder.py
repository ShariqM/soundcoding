import tensorflow as tf
import pdb
import numpy as np
import matplotlib.pyplot as plt

# Network Parameters
N = 1000 # Dim of I
n_input = 300
n_filters = 1
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
        self.P = tf.Variable(init_weights(n_filter_width, n_filters), name="Psi")
        self.T = tf.Variable(init_weights(n_filter_width, n_filters), name="Theta")
        self.tau = 16 # 64/4 ~= 16 (e-4 is about irrelevant so dt / tau = 4; tau = dt/4;)
        self.threshold = 2 # XXX ??? XXX

    def spike_activation(self, x):
        cond = tf.less(x, tf.ones(tf.shape(x)) * self.threshold)
        out = tf.select(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
        return out

    def voltage(self, x):
        cond = tf.less(x, tf.ones(tf.shape(x)) * self.threshold)
        out = tf.select(cond, x, tf.zeros(tf.shape(x)))
        return out

    def step(self, x_input, v_past):
        with tf.name_scope('neurons'):
            w_t = tf.nn.relu(tf.einsum("i,ij->j", x_input, self.P))
            v_dum_t = w_t + tf.exp(-1/self.tau) * v_past # dummy value

            v_t = self.voltage(v_dum_t)
            a_t = self.spike_activation(v_dum_t)

            #tf.scalar_summary(['a_m'] * n_filters, a_m)

        return v_t, a_t

#batch = np.random.randn(batch_size, n_input, 1)
data = np.random.randn(n_input)

with tf.Graph().as_default(), tf.Session() as sess:
    network = Network(n_filter_width, n_filters)
    data_ph = tf.placeholder(tf.float32, shape=[n_filter_width], name="data")
    v_past_ph = tf.placeholder(tf.float32, shape=[n_filters], name="v_past")
    v_t = np.zeros([n_filters])

    step_op = network.step(data_ph, v_past_ph)
    summary_op = tf.merge_all_summaries()
    init_op = tf.initialize_all_variables()

    summary_writer = tf.train.SummaryWriter('AEresults', sess.graph)

    sess.run(init_op)
    aResults = np.zeros((n_filters, (n_input - n_filter_width)))
    vResults = np.zeros((n_filters, (n_input - n_filter_width)))
    for t in range(n_input - n_filter_width):
        v_past = v_t
        print ("Step %d" % t)
        feed_dict = {data_ph: data[t:t+n_filter_width], v_past_ph: v_t}
        v_t, a_t = sess.run(step_op, feed_dict=feed_dict)
        aResults[:,t] = a_t
        vResults[:,t] = v_t

        #summary_str = sess.run(summary_op, feed_dict=feed_dict)
        #summary_writer.add_summary(summary_str, t)
        #summary_writer.flush()

    plt.subplot(2,1,1)
    plt.imshow(aResults, interpolation="nearest")
    plt.subplot(2,1,2)
    plt.imshow(vResults, interpolation="nearest")
    plt.show()

