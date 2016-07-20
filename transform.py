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
from optparse import OptionParser


parser = OptionParser()
parser.add_option("-s", "--signal", dest="signal_type", default="speech",
                  help="(speech, white, pink, harmonic)")
parser.add_option("-t", "--transform", dest="transform_type", default="wavelet",
                  help="(wavelet, ...)")
(opt, args) = parser.parse_args()

def get_signal(name):
    N = 2 ** 15
    Fs = 25000
    top = 8000.

    def scale_up(x):
        biggest = np.max(np.abs(x))
        return x * (top/biggest)

    if name == "white":
        white_noise = np.floor(scale_up(np.random.randn(N)))
        wavfile.write("white.wav", Fs, white_noise.astype(np.int16))
        return Fs, white_noise
    elif name == "pink":
        plot = False
        write = True
        white_noise = np.random.randn(N)
        F_white_noise = fft.fft(white_noise)

        freqs = fft.fftfreq(N, 1./Fs)
        freq_filter = 1./np.sqrt(np.abs(freqs))
        freq_filter[0] = 2 # Remove NaN
        if plot:
            plt.plot(freqs, freq_filter)
            plt.show()

        F_pink_noise = F_white_noise * freq_filter
        pink_noise = fft.ifft(F_pink_noise)
        pink_noise = np.floor(scale_up(np.real(pink_noise)))
        if plot:
            plt.plot(pink_noise)
            plt.show()
        wavfile.write("pink.wav", Fs, pink_noise.astype(np.int16))
        return Fs, pink_noise
    elif name == "speech":
        wav_files = glob.glob('data/wavs/s1/*')
        x = np.zeros(N)
        Fs, x_raw = wavfile.read(wav_files[1])
        x[:len(x_raw)] = x_raw
        wavfile.write("sample.wav", Fs, x.astype(np.int16))
        return Fs, x
    elif name == "harmonic":
        res = 2 ** 9 + 1
        nperiods = ceil(N/res)
        x = np.zeros(N)
        #for i in range(1,4+1):
        for i in (1,2,4,8,16):
            x += np.tile(np.sin(np.linspace(-i*np.pi, i*np.pi, (res+1))), nperiods)[:N]
        return Fs, x
    else:
        raise Exception("Unknown signal type")

def get_transform(Fs, x, name="mine"):
    if name == "wavelet":
        subsample_factor = 1
        up_to = 1000

        N = len(x)
        Fx = fft.fft(x)
        freqs = fft.fftfreq(N, 1./Fs)
        wavelets = compute_wavelets(freqs)

        wc = np.zeros((wavelets.shape[0], N), dtype=complex)
        for i in range(wavelets.shape[0]):
            wc[i,:] = fft.ifft(wavelets[i,:] * Fx)
        wc = wc[:,range(0, N, subsample_factor)] # Subsample FIXME?
        return wc[:,:up_to]
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

def get_angle_diff():
    raise NotImplementedError
#plt.subplot(323)
#plt.title("Phase Diff")
#angle_diff = np.copy(angle)
#for k in range(angle.shape[0]):
    #for i in range(angle.shape[1] - 1):
        #t1 = angle[k, i]
        #t2 = angle[k, i+1]
        #if t2 < t1:
            #t2 = t2 + 2*math.pi
        #angle_diff[k, i] = t2 - t1
    #kmean = np.mean(angle_diff[k,5:-5])
    #if k in (25,55):
        #plt.plot(angle_diff[k,5:-5], label=('K=%d' % k))
    #angle_diff[k,:] = (angle_diff[k,:] - kmean) / kmean
#plt.legend()

Fs, x = get_signal(opt.signal_type)
wc = get_transform(Fs, x, opt.transform_type)

mag, phase = librosa.core.magphase(wc)
angle = np.angle(phase)

thresh = np.mean(mag) - 0.0 * (1./2) * np.std(mag)

thresh_angle = np.copy(angle)
thresh_angle[mag < thresh] = math.pi / 3
#thresh_angle_diff = np.copy(angle_diff)
#thresh_angle_diff[mag < thresh] = -5.0

def subplot(cmd, title, data, hsv=True):
    ax = plt.subplot(cmd)
    plt.title(title)
    if hsv:
        im = plt.imshow(data, cmap=plt.get_cmap('hsv'), interpolation='none')
    else:
        im = plt.imshow(data, interpolation='none')
    plt.gca().invert_yaxis()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


plt.subplot(311)
plt.title("Input Signal")
plt.plot(x)

subplot(312, "Wavegram", mag, hsv=False)
#subplot(313, "Phase (Thresholded)", thresh_angle)
subplot(313, "Phase", angle)

plt.show()
