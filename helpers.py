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
    return np.exp(-(w - w0) ** 2 / var) * vstep(w) # FIXME Normalize?

def ln(x):
    return log(x, math.e)

def log2(x):
    return log(x, 2)

def mag_angle(wc):
    return np.absolute(wc), np.angle(wc)

def gauss_mid(mean, var):
    return math.sqrt(var * ln(2)) + mean

def var_for_bw(freq):
    bw = 1./12 # In harmonic units
    const = 2 ** (bw * 1./2) - 1
    return (2 * (freq * const) ** 2) / ln(2)


