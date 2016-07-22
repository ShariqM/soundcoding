import pdb
import sys
import numpy as np
from numpy import fft
from scipy.io import wavfile
from scipy.signal import resample
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
from math import log, ceil, pi, floor
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

    def get_start(x):
        for t in range(len(x)):
            if np.abs(x[t]) > 2000:
                return max(0, t)
        raise Exception("Signal not present?")

    def construct_signal(x_raw):
        x = np.zeros(N)
        start = get_start(x_raw)
        fill_length = min(N, len(x_raw[start:]))
        x[:fill_length] = x_raw[start:start+fill_length]
        return x

    if name == "white":
        white_noise = np.floor(scale_up(np.random.randn(N)))
        wavfile.write("test/white.wav", Fs, white_noise.astype(np.int16))
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
        wavfile.write("test/pink.wav", Fs, pink_noise.astype(np.int16))
        return Fs, pink_noise
    elif name == "speech":
        wav_files = glob.glob('data/wavs/s1/*')
        x = np.zeros(N)
        wfile = np.random.choice(wav_files)
        print "Using WAV %s" % wfile
        Fs, x_raw = wavfile.read(wfile)
        x = construct_signal(x_raw)
        wavfile.write("test/sample.wav", Fs, x.astype(np.int16))
        return Fs, x
    elif name == "harmonic":
        period = 2 ** 8
        nperiods = ceil(float(N)/period)
        x = np.zeros(N)
        #offset = np.random.random() * 2 * np.pi + 1
        offset = 0
        #for i in (1,2,4,8,16):
        for i in range(1,5+1):
            #x += np.tile(np.sin(np.linspace(-i*np.pi, i*np.pi, (period+1))), nperiods)[:N]
            #offset = np.random.randint(3) * np.pi/2
            x += np.tile(np.sin(np.linspace(-i*np.pi - offset, i*np.pi - (2*i*pi/period) - offset, period)), nperiods)[:N]
        return Fs, x
    else:
        try:
            Fs, x_raw = wavfile.read('data/mine/%s.wav' % name)
        except Exception as e:
            print "ERROR: Signal file not found"
            sys.exit(0)
        if len(x_raw.shape) > 1:
            x_raw = x_raw[:,0] # remove 2nd channel

        x = construct_signal(x_raw)
        return Fs, x

def get_transform(Fs, x, name="mine"):
    if name == "wavelet":
        subsample_factor = 1
        #start, end = 500, 1500 # For harmonic
        #start, end = 2000, 4000 # For harmonic2
        start, end = 2000, 3000 # For harmonic2
        #start, end = 3000, 4000 # speech?
        #start, end = 1000, 3000

        N = len(x)
        Fx = fft.fft(x)
        freqs = fft.fftfreq(N, 1./Fs)
        wavelets = compute_wavelets(freqs)

        wc = np.zeros((wavelets.shape[0], N), dtype=complex)
        for i in range(wavelets.shape[0]):
            wc[i,:] = fft.ifft(wavelets[i,:] * Fx)
        wc = wc[:,range(0, N, subsample_factor)] # Subsample FIXME?
        return wc[:,start:end]
    elif name == "librosa":
        import librosa
        return librosa.core.cqt(x, sr=Fs, hop_length=64, real=False)
    else:
        raise Exception("Unknown cqt type")

def mag_angle(wc):
    return np.absolute(wc), np.angle(wc)

def get_angle_diff():
    raise NotImplementedError

def all_close(x):
    for i in range(1,x.shape[0]):
        if not np.allclose(x[0], x[1], atol=1e-1):
            return False
    return True

def analysis2(mag, phase, thresh):
    def get_search_points(mag, thresh):
        points = []
        band_length = 0
        band_thresh = 1

        start = 0
        #fund_k = np.where(mag[:,0] == np.max(mag[:,0]))[0][0]
        #points.append(fund_k)
        #for start in range(fund_k, mag.shape[0]):
            #if mag[start,0] < thresh:
                #break

        for k in range(start, mag.shape[0]):
            if mag[k,0] > thresh:
                band_length = band_length + 1
            if mag[k,0] <= thresh:
                if band_length >= band_thresh:
                    points.append(k - band_length/2)
                band_length = 0
        print points
        return points


    points = get_search_points(mag, thresh)

    plt.figure(2)
    colors = ["red", "orange", "green", "blue", "purple", "black"]
    #stop = min(len(colors), len(points))
    stop = min(len(points), 4)
    for k in range(1, stop):
        i, j = points[0], points[k]
        plt.plot([-pi, pi], [0, 0], color='k')
        plt.plot([0, 0], [-pi, pi], color='k')
        plt.scatter(phase[i,:], phase[j,:], s=2, color=colors[k-1], label=("%d" % j))
        #plt.scatter(phase[i,:], phase[j,:], s=2, label=("%d" % j))

    plt.title('Phase Correlation (Fund=%d)' % points[0])
    plt.axis('equal')
    plt.legend()
    plt.show()


def analysis(mag, phase):
    t = 0
    colors = ["red", "orange", "green", "blue", "purple", "black"]
    #for t in range(1):
    #for (i,j) in ((19,31), (19, 36), (19,43), (19, 60)): # For Harmonic input
    #for (i,j) in ((13,24), (13,31), (13,35), (13,18)): # For speech
    #for (i,j) in ((14,38),(14,53),(14,62),(14,70)): # For harmonic 2
    #for (i,j) in ((15,39),(15,53),(15,62),(15,70)): # For harmonic 2
    #for (i,j) in ((25,49), (26,50), (26, 34), (26, 62), (26, 73), (26, 81)): # For speech2
    #for (i,j) in ((25,49), (25,62)): # For speech 2
    for (i,j) in ((42,66), (42,80), (42,90)): # For speech 2
        #i = np.random.randint(phase.shape[0])
        #j = np.random.randint(phase.shape[0])
        plt.plot([-pi, pi], [0, 0], color='k')
        plt.plot([0, 0], [-pi, pi], color='k')
        plt.scatter(phase[i,:], phase[j,:], s=2, color=colors[t], label=("%d" % j))
        #plt.xlabel("O_%d" % i)
        #plt.ylabel("O_%d" % j)
        #plt.axis([-pi, pi, -pi, pi])
        #plt.show()
        t = t + 1

    close_test = False
    if close_test:
        for t in range(phase.shape[1]):
            #if all_close(phase[[15,39, 53, 62, 70],t]):
            if all_close(phase[[49,62], t]):
                print "Found phase alignment", t

    plt.axis('equal')
    plt.legend()
    plt.show()

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

mag, angle = mag_angle(wc)

thresh = np.mean(mag) + 0.1 * np.std(mag)

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


plt.figure(1)
plt.subplot(311)
plt.title("Input Signal (%s)" % opt.signal_type)
plt.plot(x)

subplot(312, "Wavegram", mag, hsv=False)
#subplot(313, "Phase (Thresholded)", angle) # TODO Overlay
subplot(313, "Phase (Thresholded)", thresh_angle) # TODO Overlay
#subplot(313, "Phase", angle)

plt.show(block=False)

analysis2(mag, angle, thresh)
