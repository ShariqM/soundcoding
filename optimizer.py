import tensorflow as tf
import pdb
import numpy as np
from math import log
import matplotlib.pyplot as plt
import time
import datetime
from autoEncoder import AutoEncoder
from helpers import *
from scipy.io import wavfile
import glob

class Model():
    n_filters = 48
    n_filter_width = 512
    n_steps = 25000
    n_sample_rate = 25000
    n_batch_size = 32
    n_runs = 2 ** 10

start = datetime.datetime.now()
def wlog(stmt):
    now = datetime.datetime.now()
    print ("%s T=%ds" % (stmt, (now - start).seconds))

model = Model()

n_filters, n_filter_width = model.n_filters, model.n_filter_width
n_steps, n_batch_size = model.n_steps, model.n_batch_size
auto_encoder = AutoEncoder(n_filter_width, n_filters)

x_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_steps, 1], name="input.data")
n_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_steps, n_filters],  name="white.noise")
x_target_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_steps, 1], name="x_target")

a_ph = auto_encoder.encode(x_ph, n_ph)
x_hat_ph = auto_encoder.decode(a_ph)

cost_op = tf.reduce_mean(tf.square(x_target_ph - x_hat_ph))
init_op = tf.global_variables_initializer()

learning_rate_ph = tf.placeholder(tf.float32, shape=[])
optimizer = tf.train.GradientDescentOptimizer(learning_rate_ph).minimize(cost_op)

def get_start(x):
    largest = np.max(np.abs(x))
    for t in range(len(x)):
        if np.abs(x[t]) > 0.5 * largest:
            return max(0, t - 200) ## FIXME Make this parse arg
            #return max(0, t)

def snr(x_batch, x_hat_vals):
    R = x_batch - x_hat_vals

    var = x_batch.var().mean()
    mse = (R ** 2).mean()
    snr = 10 * log(var/mse, 10)
    return snr

plot_all = False
plot_bf  = False
#with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
with tf.Session() as sess:
    print ('Loading data')

    Fs, x_rawz = wavfile.read('data/wavs/s1/bwig3a.wav')
    start = get_start(x_rawz)
    x_raw = x_rawz[start:start + n_steps]
    x_raw = x_raw / np.max(x_raw)
    x_batch = np.reshape(np.tile(x_raw, (n_batch_size, 1)), (n_batch_size, n_steps, 1))

    noise = np.tile(np.random.randn(n_steps, n_filters), (n_batch_size, 1, 1)) * 0
    analysis_ph, synthesis_ph = auto_encoder.get_filters_ph()

    plotter = Plotter(model)
    wav_files = glob.glob('data/wavs/s1/*.wav')
    if plot_bf:
        plotter.setup_plot_bf()

    print ('Beginning runs')
    sess.run(init_op)
    for t in range(model.n_runs):
        for i in range(n_batch_size):
            #if t > 0:
                #break
            wfile = np.random.choice(wav_files)
            Fs, x_raw = wavfile.read(wfile)
            start = np.random.randint(x_raw.shape[0] - n_steps)
            x_batch[i,:,0] = x_raw[start:start+n_steps]
            x_batch[i,:,0] = x_batch[i,:,0] / np.max(x_batch[i,:,0])
            if plot_all and i == 0:
                plotter.setup_plot_3(x_batch[0,:,0])

        feed_dict = {x_ph: x_batch, n_ph: noise, x_target_ph: x_batch, \
                     learning_rate_ph: get_learning_rate(t)}

        analysis_vals, synthesis_vals, a_vals, x_hat_vals, cost, _ = \
            sess.run([analysis_ph, synthesis_ph, a_ph, x_hat_ph, cost_op, optimizer], \
                feed_dict=feed_dict)

        if t % 25 == 0:
            np.save('filters/analysis.npy', analysis_vals)
            np.save('samples/actual.npy', x_batch)
            np.save('samples/reconstruction.npy', x_hat_vals)
            print ("Files saved")

        if plot_bf and t % 5 == 0:
            print ('\tupdate')
            plotter.update_plot_bf(analysis_vals[:,0,:])
        print ("%d) Cost: %.3f, SNR: %.2fdB" % (t, cost, snr(x_batch, x_hat_vals)))
        if plot_all:
            plotter.update_plot_3(x_hat_vals[0,:,:], analysis_vals[:,0,:], synthesis_vals[:,:,0], a_vals[0,:,:])
