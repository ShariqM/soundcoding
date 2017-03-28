import numpy as np
import pdb
from numpy import fft
import matplotlib.pyplot as plt
import math
from math import log, ceil, pi, sqrt

def ln(x):
    return log(x, math.e)

def log2(x):
    return log(x, 2)

def mag_angle(wc):
    return np.absolute(wc), np.angle(wc)

def get_peaks(func):
    peaks = []
    for i in range(1, len(func) - 1):
        if func[i-1] < func[i] and func[i+1] <= func[i]:
            peaks.append(i)
    return peaks

def get_learning_rate(t):
    learning_rate = 5e-9
    bounds = [1e5 * (2 ** i) for i in range(10)]
    for bound in bounds:
        if t < bound:
            break
        learning_rate *= 0.5
        if t == bound:
            print ("Decreasing rate to: ", learning_rate)
    return learning_rate

def snr(x_batch, x_hat_vals):
    R = x_batch - x_hat_vals

    var = x_batch.var().mean()
    mse = (R ** 2).mean()
    snr = 10 * log(var/mse, 10)
    return snr


class Plotter():
    def __init__(self, model):
        self.model = model

    def setup_plot_a(self, func, target_batch):
        self.figure, axes = plt.subplots(3,2, figsize=(16,10))
        n_steps = self.model.n_steps
        n_filter_width = self.model.n_filter_width
        threshold = self.model.threshold

        axes[0,0].set_title("Signal")
        axes[0,0].plot(func[:n_steps])

        axes[1,0].set_title("Target")
        axes[1,0].set_ylim([0, 1.5])
        axes[1,0].plot(target_batch[0,:,0])

        axes[2,0].set_title("A (Network Output)")
        self.a_data = axes[2,0].plot(range(n_steps), np.ones(n_steps))[0]
        axes[2,0].set_ylim([0, 1.5])

        axes[0,1].set_title("Weights")
        axes[0,1].set_ylim([-1,1])
        self.weights_data = axes[0,1].plot(range(n_filter_width), np.ones(n_filter_width))[0]

        axes[1,1].set_title("W")
        axes[1,1].set_ylim([-1, 1])
        self.w_data = axes[1,1].plot(range(n_steps), np.zeros(n_steps))[0]

        #axes[1,1].set_title("V_dum")
        #axes[1,1].set_ylim([0, 2 * model.threshold])
        #axes[1,1].plot(range(n_steps), np.ones(n_steps) * model.threshold, linestyle='--')
        #self.v_dum_data = axes[1,1].plot(range(n_steps), np.ones(n_steps))[0]

        axes[2,1].set_title("V")
        axes[2,1].set_ylim([0, 2 * threshold])
        axes[2,1].plot(range(n_steps), np.ones(n_steps) * threshold, linestyle='--')
        self.v_data = axes[2,1].plot(range(n_steps), np.ones(n_steps))[0]

        self.figure.canvas.draw()
        plt.show(block=False)

    def update_plot_a(self, weights, w_vals, v_vals, a_vals):
        self.a_data.set_ydata(a_vals)
        self.w_data.set_ydata(w_vals)
        self.weights_data.set_ydata(weights)
        #self.weights_data.set_ydata(np.random.randn(self.model.n_filter_width))
        self.v_data.set_ydata(v_vals)
        self.figure.canvas.draw()

    def setup_plot_x(self, x_raw):
        self.figure, axes = plt.subplots(3,3, figsize=(16,12))
        n_steps = self.model.n_steps
        n_filter_width = self.model.n_filter_width
        n_filters = self.model.n_filters
        threshold = self.model.threshold


        axes[0,0].set_title("Signal")
        axes[0,0].plot(x_raw)
        axes[0,0].set_ylim([-1.5, 1.5])

        def setup_plot(loc, title, y_lim, sz, n_filters=4):
            x, y = loc[0], loc[1]
            axes[x,y].set_title(title, fontsize=18)
            axes[x,y].set_ylim(y_lim)

            data = []
            for i in range(n_filters):
                data.append(axes[x,y].plot(range(sz), np.zeros(sz))[0])
            return data

        self.x_hat_data = setup_plot((1,0), "Reconstruction", (-1.5,1.5), n_steps, 1)[0]
        self.a_data = setup_plot((2,0), "A", (0,1.5), n_steps)

        self.analysis_data = setup_plot((0,1), "Analysis", (-2,2), n_filter_width)
        self.synthesis_data = setup_plot((1,1), "Synthesis", (-1,1), n_filter_width)
        self.w_data = setup_plot((0,2), "W", (-10,10), n_steps)

        self.v_data = setup_plot((1,2), "V", (0, 2 * threshold), n_steps) # Fix threshold
        axes[1,2].plot(range(n_steps), np.ones(n_steps) * threshold, linestyle='--')

        self.figure.canvas.draw()
        plt.show(block=False)

    def update_plot_x(self, x_hat_vals, analysis_vals, synthesis_vals, w_vals, v_vals, a_vals):
        self.x_hat_data.set_ydata(x_hat_vals)
        for i in range(self.model.n_filters):
            self.a_data[i].set_ydata(a_vals[:,i])
            self.w_data[i].set_ydata(w_vals[:,i])

            self.analysis_data[i].set_ydata(analysis_vals[:,i])
            self.synthesis_data[i].set_ydata(synthesis_vals[:,i])

            self.v_data[i].set_ydata(v_vals[:,i])
        self.figure.canvas.draw()

    def setup_plot_3(self, x_raw):
        self.figure, axes = plt.subplots(3,2, figsize=(16,10))
        n_steps = self.model.n_steps
        n_filter_width = self.model.n_filter_width
        n_filters = self.model.n_filters

        axes[0,0].set_title("Signal")
        axes[0,0].plot(x_raw)
        axes[0,0].set_ylim([-1.5, 1.5])

        def setup_plot(loc, title, y_lim, sz, n_filters=4):
            x, y = loc[0], loc[1]
            axes[x,y].set_title(title, fontsize=18)
            axes[x,y].set_ylim(y_lim)

            data = []
            for i in range(n_filters):
                data.append(axes[x,y].plot(range(sz), np.zeros(sz))[0])
            return data

        self.x_hat_data = setup_plot((1,0), "Reconstruction", (-1.5,1.5), n_steps, 1)[0]
        self.a_data = setup_plot((2,0), "A", (-3,3), n_steps)

        self.analysis_data = setup_plot((0,1), "Analysis", (-1,1), n_filter_width)
        self.synthesis_data = setup_plot((1,1), "Synthesis", (-1,1), n_filter_width)

        self.figure.canvas.draw()
        plt.show(block=False)

    def update_plot_3(self, x_hat_vals, analysis_vals, synthesis_vals, a_vals):
        self.x_hat_data.set_ydata(x_hat_vals)
        for i in range(self.model.n_filters):
            self.a_data[i].set_ydata(a_vals[:,i])
            self.analysis_data[i].set_ydata(analysis_vals[:,i])
            self.synthesis_data[i].set_ydata(synthesis_vals[:,i])

        self.figure.canvas.draw()

    def setup_plot_bf2(self):
        num_rows, num_cols = 6,6

        figure, axes = plt.subplots(num_rows, num_cols, figsize=(16,10))

        k = 0
        plots = []
        for i in range(num_rows):
            for j in range(num_cols):
                plots.append(axes[i,j].plot(np.zeros(self.model.n_input))[0])
                axes[i,j].set_ylim([-0.6,0.6])
                axes[i,j].xaxis.set_visible(False)
                axes[i,j].yaxis.set_visible(False)
                k = k + 1
        self.figure = figure
        self.plots = plots
        plt.show(block=False)

    def update_plot_bf2(self, synthesis):
        n_input = synthesis.shape[0]
        for k in range(len(self.plots)):
            self.plots[k].set_data(range(n_input), synthesis[:,k])
            #self.plots[k].set_data(range(n_input), np.random.randn(n_input))
        self.figure.canvas.draw()

    def setup_plot_bf(self):
        n_filter_width = self.model.n_filter_width
        n_filters = self.model.n_filters

        sz = int(np.ceil(np.sqrt(n_filters)))
        self.figure, axes = plt.subplots(sz, sz, figsize=(16,10))
        self.data = []
        for i in range(sz):
            for j in range(sz):
                    self.data.append(axes[i,j].plot(range(n_filter_width), np.zeros(n_filter_width))[0])

        self.figure.canvas.draw()
        plt.show(block=False)

    def update_plot_bf(self, analysis_vals):
        for i in range(self.model.n_filters):
            self.data[i].set_ydata(analysis_vals[:,i])
        self.figure.canvas.draw()


''' OLD
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
'''
