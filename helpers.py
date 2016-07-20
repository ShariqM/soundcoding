import numpy as np
import pdb
from numpy import fft
import matplotlib.pyplot as plt
import math
from math import log, ceil, pi, sqrt

def step(x):
    return 1 * (x > 0)
vstep = np.vectorize(step)

def fourier_wavelet(w, w0, var):
    #return 1. / (sqrt(2 * pi * var)) * np.exp(1./2 * (w - w0) ** 2 / var) * vstep(w)
    return np.exp(-(w - w0) ** 2 / var) * vstep(w) # FIXME?

def ln(x):
    return log(x, math.e)

def log2(x):
    return log(x, 2)

def gauss_mid(mean, var):
    return math.sqrt(var * ln(2)) + mean

def var_for_bw(freq):
    const = 2 ** (1./6) - 1
    return (2 * (freq * const) ** 2) / ln(2)
    #return ((1./12) * freq) ** 2 / ln(2)
    #freq_factor = (freq * (2 ** (1./6) - 1)) ** 2
    #sq = math.sqrt(1./2)
    #return - 1./(freq_factor * ln(sq))

def compute_wavelets(freqs, plot=False):
    fmin = 2 ** 6
    noctaves = 7
    nwavelets_per_octave = 12
    nwavelets = noctaves * nwavelets_per_octave
    wavelets = np.zeros((nwavelets, len(freqs)))

    for i in range(nwavelets_per_octave * noctaves):
        w0 = fmin * 2 ** (float(i)/nwavelets_per_octave)
        var = var_for_bw(w0)
        wavelets[i,:] = (fourier_wavelet(freqs, w0, var)) ** 2
        if plot:
            plt.plot(freqs, wavelets[i,:], label=("%d" % i))
    if plot:
        sfreq = fft.fftshift(freqs)
        plt.axis([sfreq[0], sfreq[-1],  0, 1.5])
        plt.legend()
        plt.show()

    return wavelets