import tensorflow as tf
import pdb
import numpy as np
import matplotlib.pyplot as plt
import time
from autoEncoder import AutoEncoder
from helpers import *

''' Toy Data '''
class Model():
    tau_RC = 16 # 64/4 ~= 16 (e-4 is about irrelevant so dt / tau = 4; tau = dt/4;)
    tau_ABS = 24
    threshold = 0.5 # XXX ??? XXX
    n_filters = 1
    n_filter_width = 16
    n_steps = 64
    n_sample_rate = 16000
    n_batch_size = 8
    n_runs = 2 ** 10

''' Data
class Model():
    tau_RC = 160
    tau_ABS = 160
    threshold = 0.5 # XXX ??? XXX
    n_filters = 2
    n_filter_width = 160
    n_steps = 1600
    n_sample_rate = 16000
    n_batch_size = 8
    n_runs = 2 ** 10
'''

model = Model()

n_filters, n_filter_width = model.n_filters, model.n_filter_width
n_steps, n_batch_size = model.n_steps, model.n_batch_size
auto_encoder = AutoEncoder(model, n_filter_width, n_filters)

x_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_steps, n_filter_width], name="data")
v_past_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_filters], name="v_past")
a_past_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_filters], name="a_past")
f_past_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_filters], name="f_past")
a_target_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_steps, n_filters], name="a_target")
x_target_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_steps, 1], name="x_target")

def stepify(x_ph, n_filter_width, n_steps):
    x_ph = tf.transpose(x_ph, [1, 0, 2])
    x_ph = tf.reshape(x_ph, [-1, n_filter_width]) # n_batch_size * n_steps by nInput
    return tf.split(axis=0, num_or_size_splits=n_steps, value=x_ph)

def unroll(x_ph, v_past_ph, a_past_ph, f_past_ph):
    x_ph = stepify(x_ph, n_filter_width, n_steps)
    v_t_ph = v_past_ph
    a_t_ph = a_past_ph
    f_t_ph = f_past_ph

    a_ph = []
    v_dum_ph = []
    v_ph = []
    w_ph = []
    for i in range(n_steps):
        w_t_ph, v_dum_t_ph, v_t_ph, a_t_ph, f_t_ph = auto_encoder.step(x_ph[i], v_t_ph, a_t_ph, f_t_ph)
        w_ph.append(w_t_ph)
        v_dum_ph.append(v_dum_t_ph)
        v_ph.append(v_t_ph)
        a_ph.append(a_t_ph)

    return tf.transpose(tf.stack(w_ph), [1, 0, 2]), \
           tf.transpose(tf.stack(v_dum_ph), [1, 0, 2]), \
           tf.transpose(tf.stack(v_ph), [1, 0, 2]), \
           tf.transpose(tf.stack(a_ph), [1, 0, 2])

w_ph, v_dum_ph, v_ph, a_ph = unroll(x_ph, v_past_ph, a_past_ph, f_past_ph)
x_hat_ph = auto_encoder.decode(a_ph) # batch_size x n_steps x 1

#w_cost = tf.reduce_sum(w_ph)
cost_op = tf.reduce_mean(tf.reduce_sum(tf.square(x_target_ph - x_hat_ph), axis=[1,2]))

#cost_op = tf.reduce_mean(tf.reduce_sum(tf.square(a_target_ph - a_ph), axis=[1,2]))
#cost_op = tf.reduce_sum(tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(a_target_ph, a_ph))), axis=[0,1]))
#cost_op = tf.reduce_sum(tf.sqrt(tf.square(tf.subtract(a_target_ph, a_ph))))
#cost = tf.reduce_mean(tf.sqrt(tf.square(tf.sub(a_target_ph, a_ph))), reduction_indices=[0,1])

init_op = tf.global_variables_initializer()

learning_rate_ph = tf.placeholder(tf.float32, shape=[])
optimizer = tf.train.GradientDescentOptimizer(learning_rate_ph).minimize(cost_op)

make_target = False
plot_a = False
with tf.Session() as sess:
    v_t = np.zeros([n_batch_size, n_filters])
    a_t = np.zeros([n_batch_size, n_filters])
    f_t = np.zeros([n_batch_size, n_filters])

    x_raw = np.sin(np.linspace(0, 8 * np.pi, n_steps + n_filter_width))
    x_target = np.tile(x_raw[:n_steps], (n_batch_size, 1))
    x_target = x_target.reshape(n_batch_size, n_steps, 1)

    x_tmp = np.tile(x_raw, (n_batch_size, 1))
    x_batch = np.zeros((n_batch_size, n_steps, n_filter_width))
    for i in range(n_steps):
        x_batch[:,i,:] = x_tmp[:,i:i+n_filter_width]

    plotter = Plotter(model)
    #plotter.setup_plot(x_raw[n_filter_width:], target_a_batch)
    plotter.setup_plot_x(x_raw[:n_steps])

    analysis_ph, synthesis_ph = auto_encoder.get_filters_ph()

    sess.run(init_op)
    for t in range(model.n_runs):
        feed_dict = {x_ph: x_batch, v_past_ph: v_t, a_past_ph: a_t, f_past_ph: f_t, \
                     x_target_ph: x_target, learning_rate_ph: get_learning_rate(t)}

        w_vals, v_dum_vals, v_vals, a_vals, analysis_vals, synthesis_vals, x_hat_vals, cost, _ = \
            sess.run([w_ph, v_dum_ph, v_ph, a_ph, analysis_ph, synthesis_ph, x_hat_ph, cost_op, optimizer], feed_dict=feed_dict)

        print ("%d) Cost: " % (t), cost)

        #plotter.update_plot(weights, w_vals[0,:,0], v_vals[0,:,0], a_vals[0,:,0])
        plotter.update_plot_x(x_hat_vals[0,:,0], analysis_vals, synthesis_vals, w_vals[0,:,0],
                                v_vals[0,:,0], a_vals[0,:,0])
        time.sleep(0.2)

        #grad_op = tf.train.GradientDescentOptimizer(learning_rate).compute_gradients(cost_op)
        #grad_val = sess.run(grad_op, feed_dict=feed_dict)
