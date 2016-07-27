import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from helpers import *

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
    subsample_factor = 2 ** 3
    #start, end = 2000, 3000 # For harmonic2
    start, end = 3000, 5000 # For harmonic2

    N = len(x)
    Fx = fft.fft(x)
    freqs = fft.fftfreq(N, 1./Fs)
    wavelets = compute_wavelets(freqs)

    wc = np.zeros((wavelets.shape[0], N), dtype=complex)
    for i in range(wavelets.shape[0]):
        wc[i,:] = fft.ifft(wavelets[i,:] * Fx)
    wc = wc[:,range(0, N, subsample_factor)] # Subsample FIXME?
    return wc[:,start:end]


