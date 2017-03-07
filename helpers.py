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

class Plotter():
    def __init__(self, model):
        self.model = model

    def setup_plot(self, func, target_batch):
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

    def update_plot(self, weights, w_vals, v_vals, a_vals):
        self.a_data.set_ydata(a_vals)
        self.w_data.set_ydata(w_vals)
        self.weights_data.set_ydata(weights)
        #self.weights_data.set_ydata(np.random.randn(self.model.n_filter_width))
        self.v_data.set_ydata(v_vals)
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
