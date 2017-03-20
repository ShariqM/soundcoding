import tensorflow as tf
def init_weights(n_filter_width, n_filters):
    return tf.truncated_normal((n_filter_width, n_filters), 0.0,
            1/tf.sqrt(tf.cast(n_filter_width, tf.float32)))

def init_conv_weights(n_filter_width, n_filters):
    return tf.truncated_normal((n_filter_width, n_filters, 1), 0.0,
            1/tf.sqrt(tf.cast(n_filter_width, tf.float32)))

class AutoEncoder(object):
    def __init__(self, model, n_filter_width, n_filters):
        self.A = tf.Variable(init_weights(n_filter_width, n_filters), name="analysis_filters")
        self.S = tf.Variable(init_conv_weights(n_filter_width, n_filters), name="synthesis_filters")
        self.tau_RC = model.tau_RC
        self.tau_ABS = model.tau_ABS
        self.threshold = model.threshold

    def get_filters_ph(self):
        return self.A, self.S

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

    def step(self, x_input, v_past, a_past, f_past):
        with tf.name_scope('neurons'):
            w_t = tf.matmul(x_input, self.A)

            f_t = a_past + tf.exp(-1/self.tau_ABS) * f_past
            m_t = tf.nn.relu(1 - f_t)
            v_dum_t = m_t * tf.nn.relu(w_t) + tf.exp(-1/self.tau_RC) * v_past # dummy value

            v_t = self.voltage(v_dum_t)
            a_t = self.spike_activation(v_dum_t)

        return w_t, v_dum_t, v_t, a_t, f_t

    def decode(self, a_ph):
        # a_ph - Batch_size * n_steps * n_filters (in_channels = n_filters)
        x_ph = tf.nn.conv1d(a_ph, self.S, 1, padding='SAME')
        return x_ph
