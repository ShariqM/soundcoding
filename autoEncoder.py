import tensorflow as tf
import pdb
import numpy as np
import matplotlib.pyplot as plt

# Network Parameters
#n_input = 1
n_filters = 4
n_filter_width = 16
n_steps = 32
n_sample_rate = 16000
n_batch_size = 4
n_runs = 2 ** 10
learning_rate = 1e-2

def init_weights(n_filter_width, n_filters):
    return tf.truncated_normal((n_filter_width, n_filters), 0.0,
            1/tf.sqrt(tf.cast(n_filter_width, tf.float32)))

class Network(object):
    def __init__(self, n_filter_width, n_filters):
        self.P = tf.Variable(init_weights(n_filter_width, n_filters), name="Psi")
        #self.T = tf.Variable(init_weights(n_filter_width, n_filters), name="Theta")
        self.tau = 16 # 64/4 ~= 16 (e-4 is about irrelevant so dt / tau = 4; tau = dt/4;)
        self.threshold = 0.5 # XXX ??? XXX

    def spike_activation_impl(self, x):
        cond = tf.less(x, tf.ones(tf.shape(x)) * self.threshold)
        out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
        return out

    ''' Use a sigmoid gradient instead of the non-differential sign func '''
    def spike_activation(self, x):
        with tf.variable_scope('spike_activation'):
            return tf.sigmoid(x - self.threshold) + \
                tf.stop_gradient(self.spike_activation_impl(x) - tf.sigmoid(x - self.threshold))

    def voltage_impl(self, x):
        cond = tf.less(x, tf.ones(tf.shape(x)) * self.threshold)
        out = tf.where(cond, x, tf.zeros(tf.shape(x)))
        return out

    def voltage_for_grad(self, x):
        cond = tf.less(x, tf.ones(tf.shape(x)) * self.threshold)
        out = tf.where(cond, x, self.threshold * tf.exp(-(x-self.threshold)))
        return out

    def voltage(self, x):
        with tf.variable_scope('voltage'):
            return self.voltage_for_grad(x) + \
                tf.stop_gradient(self.voltage_impl(x) - self.voltage_for_grad(x))

    def step(self, x_input, v_past):
        with tf.name_scope('neurons'):
            w_t = tf.matmul(x_input, self.P)
            v_dum_t = tf.nn.relu(w_t) + tf.exp(-1/self.tau) * v_past # dummy value

            v_t = self.voltage(v_dum_t)
            a_t = self.spike_activation(v_dum_t)

        return w_t, v_dum_t, v_t, a_t

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

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_op)

make_target = False
plot_a = False
with tf.Session() as sess:
    v_t = np.zeros([n_batch_size, n_filters])

    data_tmp = np.tile(np.random.randn(n_steps + n_filter_width), (n_batch_size, 1))
    data_batch = np.zeros((n_batch_size, n_steps, n_filter_width))
    for i in range(n_steps):
        data_batch[:,i,:] = data_tmp[:,i:i+n_filter_width]

    target_batch = np.zeros((n_batch_size, n_steps, n_filters))
    target_batch[:,range(0,n_steps,8),:] = 1
    sess.run(init_op)
    for t in range(n_runs):
        #data_batch = np.random.randn(n_batch_size, n_steps, n_input)
        #target_batch = np.load('a_target.npy')

        feed_dict = {data_ph: data_batch, v_past_ph: v_t, target_ph: target_batch}
        cost = sess.run(cost_op, feed_dict=feed_dict)
        print ("%d) Cost: " % (t), cost)

        #grad_op = tf.train.GradientDescentOptimizer(learning_rate).compute_gradients(cost_op)
        #grad_val = sess.run(grad_op, feed_dict=feed_dict)

        sess.run(optimizer, feed_dict=feed_dict)

        if make_target:
            print ('target made')
            a = sess.run(a_ph, feed_dict=feed_dict)
            np.save('a_target', a)
            make_target = False


        def plotter(op, title, plot_num):
            r = sess.run(op, feed_dict=feed_dict)
            plt.subplot(4,1,plot_num)
            plt.title(title)
            plt.plot(r[0,:,0])

        if plot_a:
            plotter(w_ph, "W", 1)
            plotter(v_ph, "V", 2)
            plotter(a_ph, "A", 3)

            plt.title("Target")
            plt.subplot(4,1,4)
            plt.plot(target_batch[0,:,0])

            plt.show()
