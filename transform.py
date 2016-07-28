import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from helpers import *

def step(x):
    return 1 * (x > 0)
vstep = np.vectorize(step)

def fourier_wavelet(w, w0, var):
    return np.exp(-(w - w0) ** 2 / var) * vstep(w) # FIXME Normalize?

def var_for_bw(freq):
    bw = 1./12 # In harmonic units
    const = 2 ** (bw * 1./2) - 1
    return (2 * (freq * const) ** 2) / ln(2)

def compute_wavelets(freqs, plot=False):
    fmin = 2 ** 6
    noctaves = 5
    nwavelets_per_octave = 24
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

def get_transform(Fs, x):
    subsample_factor = 2 ** 5
    #start, end = 2000, 3000 # For harmonic2
    start, end = 0000, 32000 # For harmonic2

    N = len(x)
    Fx = fft.fft(x)
    freqs = fft.fftfreq(N, 1./Fs)
    wavelets = compute_wavelets(freqs)

    wc = np.zeros((wavelets.shape[0], N), dtype=complex)
    for i in range(wavelets.shape[0]):
        wc[i,:] = fft.ifft(wavelets[i,:] * Fx)
    wc = wc[:,range(0, N, subsample_factor)] # Subsample FIXME?
    return wc[:,start:end]


