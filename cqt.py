import pdb
import numpy as np
from numpy import fft
from scipy.io import wavfile
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import librosa
import math
from math import log, ceil, pi
from helpers import *

signal_type = "speech"
cqt_type = "wavelet"

def get_signal(name):
    def_Fs = 25000
    if name == "noise":
        return def_Fs, np.random.randn(30000)
    elif name == "speech":
        wav_files = glob.glob('data/wavs/s1/*')
        Fs, x = wavfile.read(wav_files[1])
        return Fs, x
    elif name == "harmonic":
        res = 2 ** 9 + 1
        nperiods = ceil(def_Fs/res * 0.25)
        x = np.zeros((res+1) * nperiods)
        #for i in range(1,4+1):
        for i in (1,2,4,8,16):
            x += np.tile(np.sin(np.linspace(-i*np.pi, i*np.pi, (res+1))), nperiods)
        return def_Fs, x
    else:
        raise Exception("Unknown signal type")

def get_cqt(Fs, x, name="mine"):
    if name == "wavelet":
        N = 2 ** 13
        x = x[:N]
        Fx = fft.fft(x)
        freqs = fft.fftfreq(N, 1./Fs)
        wavelets = compute_wavelets(freqs)

        wc = np.zeros((wavelets.shape[0], N), dtype=complex)
        for i in range(wavelets.shape[0]):
            wc[i,:] = fft.ifft(wavelets[i,:] * Fx)
        wc = wc[:,range(0,N,2**4)] # Subsample FIXME?
        return wc
    elif name == "mine":
        hop_length = 64
        b = 12 # Filters per octave
        Fmin = 80
        Fmax = Fmin * 2 ** 4

        K = int(ceil(b * log(Fmax/Fmin, 2))) # Number of cq bins
        Q = (2 ** (1./b) - 1) ** (-1)


        f = np.zeros(K+1)
        N = np.zeros(K+1)
        timepoints = int(ceil(x.shape[0] / hop_length))
        cq = np.zeros((K+1, timepoints * 2), dtype=complex)
        print "Fs=%f, Fmin=%f, Fmax=%f, K=%d, Q=%f" % (Fs, Fmin, Fmax, K, Q)
        #print x.shape[0]

        dummy_f = Fmin * 2 ** (1./b)
        dummy_Nk = int(ceil(Q * Fs/dummy_f))
        #x = np.lib.pad(x, (dummy_Nk, dummy_Nk), 'constant', constant_values=(0,0))
        x = np.lib.pad(x, (dummy_Nk/4, dummy_Nk), 'constant', constant_values=(0,0))

        t = 0
        for s in range(0, x.shape[0] - dummy_Nk, hop_length):
            for k in range(1,K+1):
                f[k] = Fmin * 2 ** (float(k)/b)
                N[k] = ceil(Q * Fs/f[k])
                Nk = int(N[k])
                ns = np.array(range(Nk))
                #print Nk

                e = np.exp(-2 * pi * 1j * (float(Q)/N[k]) * ns)
                h = np.hamming(N[k]) * e
                cq[k,t] = float(1)/Nk * np.dot(x[s:(s+Nk)], h)
            t = t + 1
        return cq
    elif name == "librosa":
        return librosa.core.cqt(x, sr=Fs, hop_length=64, real=False)
    else:
        raise Exception("Unknown cqt type")



Fs, x = get_signal(signal_type)
plt.subplot(321)
plt.title("Input Signal")
plt.plot(x)
cqt = get_cqt(Fs, x, cqt_type)

mag, phase = librosa.core.magphase(cqt)
angle = np.angle(phase)
thresh = np.mean(mag) - 0.0* (1./2) * np.std(mag)
thresh_angle = np.copy(angle)
thresh_angle[mag < thresh] = math.pi / 3

#for k in (23, 24, 34, 35, 36, 37):
plt.subplot(323)
plt.title("Phase Diff")
angle_diff = np.copy(angle)
for k in range(angle.shape[0]):
    for i in range(angle.shape[1] - 1):
        t1 = angle[k, i]
        t2 = angle[k, i+1]
        if t2 < t1:
            t2 = t2 + 2*math.pi
        angle_diff[k, i] = t2 - t1
        #print angle_diff[k, i], t1, t2
    kmean = np.mean(angle_diff[k,5:-5])
    #if k in (10,23,30):
    #if k in (6,14,18):
    #if k in range(2,7):
    #if k in (6,14):
    #if k in (6,19,31,42):
    #if k in (7,19,31,43):
    #if k in (8,20,31,14):
    #if k in (25,37,):
    #if k in (10,20,29):
    if k in (10,20):
    #if k in (3,15,27,39):
        #plt.plot(angle_diff[k,5:-5], label=('K=%d' % k))
        plt.plot(angle[k,5:-5], label=('K=%d' % k))
    angle_diff[k,:] = (angle_diff[k,:] - kmean) / kmean
    #if k in (6,):
plt.legend()
#plt.plot((angle_diff - np.mean(angle_diff)) / np.mean(angle_diff))
#plt.legend()
thresh_angle_diff = np.copy(angle_diff)
thresh_angle_diff[mag < thresh] = -5.0


ax = plt.subplot(322)
plt.title("CQT")
im = ax.imshow(mag, interpolation='none')
plt.gca().invert_yaxis()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)

ax = plt.subplot(324)
plt.title("Phase (Thresholded)")
im = plt.imshow(thresh_angle, cmap=plt.get_cmap('hsv'), interpolation='none')
plt.gca().invert_yaxis()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)

ax = plt.subplot(325)
plt.title("Phase")
im = plt.imshow(angle, cmap=plt.get_cmap('hsv'), interpolation='none')
plt.gca().invert_yaxis()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)

ax = plt.subplot(326)
plt.title("Phase Diff (Threshold)")
vthresh=1.00
im = ax.imshow(thresh_angle_diff, interpolation='none', vmin=-vthresh, vmax=vthresh)
plt.gca().invert_yaxis()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)



plt.show()
