import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from scipy.io import wavfile
from helpers import *
from math import acos, sqrt, pi
from common import *
import pdb

def step(x):
    return 1 * (x > 0)
vstep = np.vectorize(step)

def gaussian_wavelet(w, w0, var):
    return np.exp(-(w - w0) ** 2 / var) * vstep(w) # FIXME Normalize?

def sinusoid_wavelet(w, w0, var):
    arg = (w - w0) / var
    arg[arg <= -pi] = -pi
    arg[arg >=  pi] =  pi
    return (np.cos(arg) + 1.) / 2

def gaussian_var(freq, bandwidth):
    bw = 1./bandwidth # In harmonic units
    const = 2 ** (bw * 1./2) - 1
    return ((freq * const) ** 2) / ln(2)

def sinusoid_var(freq, freq_prev):
    return 2 * (freq - freq_prev) / pi

def compute_wavelets(freqs, opt, plot=False):
    fmin = 2 ** 6
    noctaves = 7
    nwavelets_per_octave = opt.nfilters_per_octave
    nwavelets = noctaves * nwavelets_per_octave
    wavelets = np.zeros((nwavelets, len(freqs)))

    if plot:
        plt.figure()

    total = np.zeros(len(freqs))
    for i in range(nwavelets_per_octave * noctaves):
        w0 = fmin * 2 ** (float(i)/nwavelets_per_octave)

        if opt.wavelet_type == "gaussian":
            var = gaussian_var(w0, opt.bandwidth)
            wavelets[i,:] = (gaussian_wavelet(freqs, w0, var)) ** 2
        else:
            w_prev = fmin * 2 ** (float(i-1)/nwavelets_per_octave)
            w_next = fmin * 2 ** (float(i+1)/nwavelets_per_octave)
            var = sinusoid_var(w_next, w0)
            #var = sinusoid_var(w0, w_prev)
            wavelets[i,:] = (sinusoid_wavelet(freqs, w0, var))

        if plot:
            plt.plot(freqs, wavelets[i,:], label=("%d" % i))
        #total += wavelets[i,:]
        total += wavelets[i,:] ** 2

    if plot:
        sfreq = fft.fftshift(freqs)
        plt.axis([sfreq[0], sfreq[-1],  0, 1.5])
        plt.legend()
        plt.show()

    if opt.plot_total:
        plt.figure()
        plt.title("Total")
        top = fmin * 2 ** (noctaves+1)
        plt.plot(range(top), total[:top])
        plt.show()

    return wavelets

def transform(Fs, x, opt, do_plot=False):
    subsample_factor = 2 ** opt.subsample_power
    start, end = 0, opt.time_length

    N = len(x)
    Fx = fft.fft(x)
    freqs = fft.fftfreq(N, 1./Fs)
    if do_plot:
        plot_fourier(freqs, Fx, 100)
    wavelets = compute_wavelets(freqs, opt, plot=False)

    wc = np.zeros((wavelets.shape[0], N), dtype=complex)
    for i in range(wavelets.shape[0]):
        wc[i,:] = fft.ifft(wavelets[i,:] * Fx)

    wc = wc[:,range(0, N, subsample_factor)]
    return wc[:,start:end]

def itransform(wc, Fs, opt):
    N = opt.time_length
    Fx_recon = np.zeros(N, dtype=complex)
    freqs = fft.fftfreq(N, 1./Fs)
    wavelets = compute_wavelets(freqs, opt)

    for i in range(wavelets.shape[0]):
        Filx = fft.fft(wc[i,:])
        Fx_recon += Filx * wavelets[i,:]

    x_recon = np.real(fft.ifft(Fx_recon)) # XXX Real OK?
    scale_factor = opt.MAX_AMP/np.max(x_recon)
    print '\tScaling Reconstruction by %.2f' % scale_factor
    x_recon *= scale_factor
    return x_recon
