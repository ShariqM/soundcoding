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
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-l", "--load_filters", action='store_true', dest="load",
                  default=False)
(opt, args) = parser.parse_args()

class Model():
    n_filters = 32
    n_filter_width = 128
    n_steps = 2 * 14700
    n_batch_size = 24
    n_runs = 2 ** 10
    Lambda = 0

start = datetime.datetime.now()
def wlog(stmt):
    now = datetime.datetime.now()
    print ("%s T=%ds" % (stmt, (now - start).seconds))

model = Model()

n_filters, n_filter_width = model.n_filters, model.n_filter_width
n_steps, n_batch_size = model.n_steps, model.n_batch_size
auto_encoder = AutoEncoder(model)

x_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_steps, 1], name="input.data")
n_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_steps, n_filters],  name="white.noise")
x_target_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_steps, 1], name="x_target")

u_var_ph, r_ph = auto_encoder.encode(x_ph, n_ph)
x_hat_ph = auto_encoder.decode(r_ph)

cost_op = tf.reduce_mean(tf.square(x_target_ph - x_hat_ph)) + model.Lambda * tf.reduce_mean(r_ph)
init_op = tf.global_variables_initializer()
analysis_ph, synthesis_ph = auto_encoder.get_filters_ph()
norm_a_op = tf.nn.l2_normalize(analysis_ph, 0)
norm_s_op = tf.nn.l2_normalize(synthesis_ph, 0)

learning_rate_ph = tf.placeholder(tf.float32, shape=[])
optimizer = tf.train.GradientDescentOptimizer(learning_rate_ph).minimize(cost_op)

plot_all = False
plot_bf  = False
#with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
with tf.Session() as sess:
    sess.run(init_op)
    if opt.load:
        print ('loading weights')
        assign_analysis = analysis_ph.assign(np.load('filters/analysis.npy'))
        assign_synthesis = synthesis_ph.assign(np.load('filters/synthesis.npy'))
        sess.run(assign_analysis)
        sess.run(assign_synthesis)

    noise = np.tile(np.random.randn(n_steps, n_filters), (n_batch_size, 1, 1)) * 0
    plotter = Plotter(model)
    wav_files = glob.glob('data/lewicki_audiodata/mammals/*.wav')
    if plot_bf:
        plotter.setup_plot_bf()

    print ('Beginning runs')
    x_batch = np.zeros((n_batch_size, n_steps, 1))
    for t in range(model.n_runs):
        for i in range(n_batch_size):
            #if t > 0:
                #break
            wfile = np.random.choice(wav_files)
            Fs, x_raw = wavfile.read(wfile)
            start = np.random.randint(x_raw.shape[0] - n_steps)
            x_batch[i,:,0] = x_raw[start:start+n_steps]

            x_batch[i,:,0] = x_batch[i,:,0] / (np.max(x_batch[i,:,0]) / 10)
            #norm = np.covariance(x_batch[i,:,0])
            #for j in range(n_steps):
                #x_batch[i,j,0] += np.random.normal(0.0, 0.2)
            #x_batch[i,:,0] = x_batch[i,:,0] / np
            if plot_all and i == 0:
                plotter.setup_plot_3(x_batch[0,:,0])

        feed_dict = {x_ph: x_batch, n_ph: noise, x_target_ph: x_batch, \
                     learning_rate_ph: get_learning_rate(t)}

        analysis_vals, synthesis_vals, r_vals, x_hat_vals, cost, _ = \
            sess.run([analysis_ph, synthesis_ph, r_ph, x_hat_ph, cost_op, optimizer], \
                feed_dict=feed_dict)

        if t % 25 == 0:
            if t > 0:
                np.save('record/filters/analysis.npy', analysis_vals)
                np.save('record/filters/synthesis.npy', synthesis_vals)
                print ('\t Weights Saved')
            np.save('record/samples/actual.npy', x_batch)
            np.save('record/samples/reconstruction.npy', x_hat_vals)
            np.save('record/activity.npy', r_vals)
            print ("Samples saved | Mean(r)=%.4f" % np.mean(r_vals))

        if plot_bf and t % 5 == 0:
            print ('\tupdate')
            plotter.update_plot_bf(analysis_vals[:,0,:])
        print ("%d) Cost: %.3f, SNR: %.2fdB" % (t, cost, snr(x_batch, x_hat_vals)))
        if plot_all:
            plotter.update_plot_3(x_hat_vals[0,:,:], analysis_vals[:,0,:], synthesis_vals[:,:,0], r_vals[0,:,:])
