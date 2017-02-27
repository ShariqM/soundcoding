import tensorflow as tf
import pdb
import numpy as np
import matplotlib.pyplot as plt

# Network Parameters
N = 1000 # Dim of I
n_input = 300
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
        self.P = tf.Variable(init_weights(n_filter_width, n_filters), name="Psi")
        self.T = tf.Variable(init_weights(n_filter_width, n_filters), name="Theta")

    def step(self, x_input):
        #v_m = tf.conv1d(x_input, self.F, 1, 'SAME') # Add leak?
        # Add Noise?
        with tf.name_scope('neurons'):

            v_rest  = tf.constant([0] * n_filters, name='v_rest', dtype=tf.float32)
            a_rest  = tf.constant([0] * n_filters, name='v_rest', dtype=tf.float32)
            a_spike = tf.constant([1] * n_filters, name='a_spike', dtype=tf.float32)

            # Trainable?
            #import pdb
            #pdb.set_trace()
            #t = t[0]

            #with tf.variable_scope("time") as scope:
            #v_m = tf.Variable(tf.zeros(n_filters), trainable=True, name="v_m")
            #a_m = tf.Variable(tf.zeros(n_filters), trainable=True, name="a_m")
                #scope.reuse_variables()

            v_conv = tf.einsum("i,ij->j", x_input, self.F)
# Multiply by e^(-t/tau)
            a_m =
            #v_conv = tf.reduce_sum(tf.mul(x_input, F)) # Dot product
            #v_conv = tf.conv1d(x_input, self.F, 1, 'SAME')
            #tf.scalar_summary(['a_m'] * n_filters, a_m)

            def spike_rest_op():
                return tf.tuple((v_m.assign(v_rest),
                                 a_m.assign(a_spike)))

            def responding_op():
                return tf.tuple((v_m.assign(v_conv), a_spike,
                                a_m))

            threshold = 2.0
            a_m = tf.select(tf.greater(v_conv, threshold), a_spike, a_rest)
            v_m = tf.select(tf.greater(v_conv, threshold), v_rest, v_conv)
            #_step = tf.case((
                             #(tf.reshape(tf.greater(v_m, threshold), [n_filters]), spike_rest_op),
                             #(tf.greater(v_m, threshold), spike_rest_op),
                            #),
                             #responding_op)
        #return tf.group(a_m, v_m)
        return (a_m, v_m)

#batch = np.random.randn(batch_size, n_input, 1)
data = np.random.randn(n_input)

with tf.Graph().as_default(), tf.Session() as sess:
    network = Network(n_filter_width, n_filters)
    dataPH = tf.placeholder(tf.float32, shape=[n_filter_width], name="data")
    #tPH = tf.placeholder(tf.int32, shape=[1], name="t")

    step_op = network.step(dataPH)
    summary_op = tf.merge_all_summaries()
    init_op = tf.initialize_all_variables()

    summary_writer = tf.train.SummaryWriter('AEresults', sess.graph)

    sess.run(init_op)
    aResults = np.zeros((n_filters, (n_input - n_filter_width)))
    vResults = np.zeros((n_filters, (n_input - n_filter_width)))
    for t in range(n_input - n_filter_width):
        print ("Step %d" % t)
        #feed_dict = {tPH: [t], dataPH: data[t:t+n_filter_width]}
        feed_dict = {dataPH: data[t:t+n_filter_width]}
        a_m, v_m = sess.run(step_op, feed_dict=feed_dict)
        aResults[:,t] = a_m
        vResults[:,t] = v_m

        #summary_str = sess.run(summary_op, feed_dict=feed_dict)
        #summary_writer.add_summary(summary_str, t)
        #summary_writer.flush()

    plt.subplot(2,1,1)
    plt.imshow(aResults, interpolation="nearest")
    plt.subplot(2,1,2)
    plt.imshow(vResults, interpolation="nearest")
    plt.show()

