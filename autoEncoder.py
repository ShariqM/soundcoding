import tensorflow as tf
import pdb
import numpy as np
import matplotlib.pyplot as plt
import time
from network import Network
from helpers import *

# Network Parameters
#n_input = 1
n_filters = 1
n_filter_width = 16
n_steps = 64
n_sample_rate = 16000
n_batch_size = 8
n_runs = 2 ** 10

network = Network(n_filter_width, n_filters)

data_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_steps, n_filter_width], name="data")
v_past_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_filters], name="v_past")
target_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_steps, n_filters], name="a_target")

def stepify(x_ph, n_filter_width, n_steps):
    x_ph = tf.transpose(x_ph, [1, 0, 2])
    x_ph = tf.reshape(x_ph, [-1, n_filter_width]) # n_batch_size * n_steps by nInput
    return tf.split(axis=0, num_or_size_splits=n_steps, value=x_ph)

def unroll(data_ph, v_past_ph):
    x_ph = stepify(data_ph, n_filter_width, n_steps)
    v_t_ph = v_past_ph

    a_ph = []
    v_dum_ph = []
    v_ph = []
    w_ph = []
    for i in range(n_steps):
        w_t_ph, v_dum_t_ph, v_t_ph, a_t_ph = network.step(x_ph[i], v_t_ph)
        w_ph.append(w_t_ph)
        v_dum_ph.append(v_dum_t_ph)
        v_ph.append(v_t_ph)
        a_ph.append(a_t_ph)

    return tf.transpose(tf.stack(w_ph), [1, 0, 2]), \
           tf.transpose(tf.stack(v_dum_ph), [1, 0, 2]), \
           tf.transpose(tf.stack(v_ph), [1, 0, 2]), \
           tf.transpose(tf.stack(a_ph), [1, 0, 2])

w_ph, v_dum_ph, v_ph, a_ph = unroll(data_ph, v_past_ph)
w_cost = tf.reduce_sum(w_ph)
cost_op = tf.reduce_mean(tf.reduce_sum(tf.square(target_ph - a_ph), axis=[1,2]))
#cost_op = tf.reduce_sum(tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(target_ph, a_ph))), axis=[0,1]))
#cost_op = tf.reduce_sum(tf.sqrt(tf.square(tf.subtract(target_ph, a_ph))))
#cost = tf.reduce_mean(tf.sqrt(tf.square(tf.sub(target_ph, a_ph))), reduction_indices=[0,1])
init_op = tf.global_variables_initializer()

def get_learning_rate(t):
    if t < 30:
        return 1e-3
    if t < 80:
        return 1e-4
    if t < 120:
        return 1e-5
    return 1e-6

learning_rate = tf.placeholder(tf.float32, shape=[])
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_op)

make_target = False
plot_a = False
with tf.Session() as sess:
    v_t = np.zeros([n_batch_size, n_filters])

    func = np.zeros(n_steps + n_filter_width)
    func[n_filter_width:] = np.sin(np.linspace(0, 8 * np.pi, n_steps))
    peaks = get_peaks(func[n_filter_width:])
    print ("Peaks", peaks)

    data_tmp = np.tile(func, (n_batch_size, 1))
    data_batch = np.zeros((n_batch_size, n_steps, n_filter_width))
    for i in range(n_steps):
        data_batch[:,i,:] = data_tmp[:,i:i+n_filter_width]

    target_batch = np.zeros((n_batch_size, n_steps, n_filters))
    target_batch[:,peaks,:] = 1

    plotter = Plotter(n_steps)
    plotter.setup_plot(func[n_filter_width:], target_batch)

    sess.run(init_op)
    for t in range(n_runs):
        #data_batch = np.random.randn(n_batch_size, n_steps, n_input)
        #target_batch = np.load('a_target.npy')

        feed_dict = {data_ph: data_batch, v_past_ph: v_t, target_ph: target_batch, \
                     learning_rate: get_learning_rate(t)}
        w_vals, v_dum_vals, v_vals, a_vals, cost, _ = sess.run([w_ph, v_dum_ph, v_ph, a_ph, \
                                                        cost_op, optimizer], feed_dict=feed_dict)
        print ("%d) Cost: " % (t), cost)

        plotter.update_plot(w_vals[0,:,0], v_dum_vals[0,:,0], v_vals[0,:,0], a_vals[0,:,0])
        time.sleep(0.2)

        #grad_op = tf.train.GradientDescentOptimizer(learning_rate).compute_gradients(cost_op)
        #grad_val = sess.run(grad_op, feed_dict=feed_dict)
