import numpy as np
import math
from math import log, ceil, pi, sqrt

def step(x):
    return 1 * (x > 0)
vstep = np.vectorize(step)

def fourier_wavelet(w, w0, var):
    #return 1. / (sqrt(2 * pi * var)) * np.exp(1./2 * (w - w0) ** 2 / var) * vstep(w)
    return 1. / (pi ** (1./4)) * np.exp(-(w - w0) ** 2 / var) * vstep(w) # FIXME?

def ln(x):
    return log(x, math.e)

def log2(x):
    return log(x, 2)

def gauss_mid(mean, var):
    return math.sqrt(var * ln(2)) + mean

def var_for_bw(octave_size):
    return ((1./8) * octave_size) ** 2 / ln(2)

def compute_wavelets(freqs):
    fmin = 80
    noctaves = 4
    nwavelets_per_octave = 32
    nwavelets = noctaves * nwavelets_per_octave
    wavelets = np.zeros((nwavelets, len(freqs)))

    for i in range(nwavelets_per_octave * noctaves):
        w0 = fmin * 2 ** (float(i)/nwavelets_per_octave)
        var = var_for_bw(w0)
        wavelets[i,:] = fourier_wavelet(freqs, w0, var)

    return wavelets
