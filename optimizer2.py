import tensorflow as tf
import pdb
import numpy as np
from math import log
import matplotlib.pyplot as plt
import time
import datetime
from autoEncoder2 import AutoEncoder2
from helpers import *
from scipy.io import wavfile
import glob
from optparse import OptionParser
from common_ica import construct_data

parser = OptionParser()
parser.add_option("-l", "--load_filters", action='store_true', dest="load",
                  default=False)
(opt, args) = parser.parse_args()

class Model():
    n_input = 128
    n_filters = 128
    n_batch_size = 640
    n_runs = 2 ** 12
    Lambda = 4000.0

start = datetime.datetime.now()
def wlog(stmt):
    now = datetime.datetime.now()
    print ("%s T=%ds" % (stmt, (now - start).seconds))

model = Model()

n_input, n_filters, n_batch_size, n_runs = model.n_input, model.n_filters, model.n_batch_size, model.n_runs
auto_encoder = AutoEncoder2(model)

x_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_input], name="input.data")
n_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_filters],  name="white.noise")
x_target_ph = tf.placeholder(tf.float32, shape=[n_batch_size, n_input,], name="x_target")

u_ph = auto_encoder.encode(x_ph, n_ph)
x_hat_ph = auto_encoder.decode(u_ph)

cost_op = tf.reduce_sum(tf.reduce_mean(tf.square(x_target_ph - x_hat_ph), axis=0)) + model.Lambda * tf.reduce_sum(tf.reduce_mean(tf.abs(u_ph), axis=0))
#cost_op = tf.reduce_sum(tf.reduce_mean(tf.square(x_target_ph - x_hat_ph), axis=0)) + model.Lambda * tf.reduce_sum(tf.reduce_mean(tf.log(1 + tf.square(u_ph)), axis=0))
init_op = tf.global_variables_initializer()
analysis_ph, synthesis_ph = auto_encoder.get_filters_ph()
norm_a_op = analysis_ph.assign(tf.nn.l2_normalize(analysis_ph, 1))
norm_s_op = synthesis_ph.assign(tf.nn.l2_normalize(synthesis_ph, 0))

learning_rate_ph = tf.placeholder(tf.float32, shape=[])
optimizer = tf.train.GradientDescentOptimizer(learning_rate_ph).minimize(cost_op)

#def whiten(X):
    #X -= np.mean(X, axis=0)
    #cov = np.dot(X.T, X) / X.shape[0]
    #U,S,V = np.linalg.svd(cov)
    #X_rot = np.dot(X, U)
    #X_white = X_rot / np.sqrt(S + 1e-8)
    #return X_white, U, S

N = 1000
data = construct_data("mammals", N, n_input)
#data, U, S = whiten(data)

plot_all = False
plot_bf  = True
#with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
with tf.Session() as sess:
    sess.run(init_op)
    if opt.load:
        print ('loading weights')
        assign_analysis = analysis_ph.assign(np.load('record/filters2/analysis.npy'))
        assign_synthesis = synthesis_ph.assign(np.load('record/filters2/synthesis.npy'))
        sess.run(assign_analysis)
        sess.run(assign_synthesis)

    noise = np.zeros((n_batch_size, n_filters))
    plotter = Plotter(model)
    wav_files = glob.glob('data/lewicki_audiodata/mammals/*.wav')
    if plot_bf:
        plotter.setup_plot_bf2()

    print ('Beginning runs')
    x_batch = np.zeros((n_batch_size, n_input))
    for t in range(model.n_runs):
        for i in range(n_batch_size):
            k = np.random.randint(data.shape[0])
            x_batch[i,:] = data[k,:]

        feed_dict = {x_ph: x_batch, n_ph: noise, x_target_ph: x_batch, \
                     learning_rate_ph: get_learning_rate(t)}

        analysis_vals, synthesis_vals, u_vals, x_hat_vals, cost, _ = \
            sess.run([analysis_ph, synthesis_ph, u_ph, x_hat_ph, cost_op, optimizer], \
                feed_dict=feed_dict)

        #sess.run([norm_a_op, norm_s_op])
        sess.run(norm_s_op)

        if t % 25 == 0:
            if t > 0:
                #analysis_vals = # n_filters * n_input
                #analysis_vals = np.dot((np.sqrt(S) + 1e-8) * analysis_vals, U)
                #synthesis_vals = np.dot(U, ((np.sqrt(S) + 1e-8) * synthesis_vals.T).T)
                np.save('record/filters2/analysis.npy', analysis_vals)
                np.save('record/filters2/synthesis.npy', synthesis_vals)
                print ('\t Weights Saved')
            np.save('record/samples2/actual.npy', x_batch)
            np.save('record/samples2/reconstruction.npy', x_hat_vals)
            np.save('record/activity2.npy', u_vals)
            #print ("Samples saved | Mean(u)=%.2f%%" % (100 * (np.sum(np.abs(u_vals)) / (n_batch_size * n_input))))
            print ("Samples saved | Mean(u)=%.2f" % (np.sum(np.mean(np.abs(u_vals), axis=0))))
            #print ("Samples saved | Mean(u)=%.2f" % (np.sum(np.mean(np.log(1 + u_vals ** 2), axis=0))))

        if plot_bf and t % 25 == 0:
            plotter.update_plot_bf2(synthesis_vals)
        if t % 5 == 0:
            print ("%d) Cost: %.3f, SNR: %.2fdB" % (t, cost, snr(x_batch, x_hat_vals)))
        if plot_all:
            plotter.update_plot_3(x_hat_vals[0,:,:], analysis_vals[:,0,:], synthesis_vals[:,:,0], r_vals[0,:,:])
