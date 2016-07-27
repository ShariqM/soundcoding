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
from scipy import stats


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
        largest = np.max(np.abs(x))
        for t in range(len(x)):
            if np.abs(x[t]) > 0.1 * largest:
                return max(0, t - 100)
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
        wav_files = glob.glob('data/wavs/s1_50kHz/*')
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
                section = mag[k-band_length:k,0]
                biggest_k = (k-band_length) + np.where(section == np.max(section))[0][0]
                points.append(biggest_k)
                #points.append(k - band_length/2)
            band_length = 0
    print points
    return points


def analysis(mag, angle, thresh):
    points = get_search_points(mag, thresh)

    plt.figure(2)
    colors = ["red", "orange", "green", "blue", "purple", "black"]
    #stop = min(len(colors), len(points))
    stop = min(len(points), 4)
    for k in range(1, stop):
        i, j = points[0], points[k]
        plt.plot([-pi, pi], [0, 0], color='k')
        plt.plot([0, 0], [-pi, pi], color='k')
        plt.scatter(angle[i,:], angle[j,:], s=2, color=colors[k-1], label=("%d" % j))
        #plt.scatter(phase[i,:], phase[j,:], s=2, label=("%d" % j))

    plt.title('Phase Correlation (Fund=%d)' % points[0])
    plt.axis('equal')
    plt.legend()
    plt.show()

def phase_diff(mag, angle, thresh):
    angle_diffs = np.zeros((angle.shape[0], angle.shape[1] - 1))
    total_angle = np.zeros((angle.shape[0], angle.shape[1] - 1))
    for k in range(angle.shape[0]):
        for i in range(angle.shape[1] - 1):
            a = angle[k,i]
            b = angle[k,i+1]
            if b < a:
                b = b + 2*pi
            angle_diffs[k,i] = (b - a)
            if i > 0:
                total_angle[k,i] = total_angle[k,i-1] + (b-a)
            else:
                total_angle[k,i] = a
        #print np.mean(angle_diffs[k]), np.std(angle_diffs[k])

    points = get_search_points(mag, thresh)
    i,j = points[6], points[9]

    #i,j = 13, 37
    #i,j = 29, 53
    x,y = total_angle[i,:], total_angle[j,:]
    a, b, r_value, c, std_err = stats.linregress(x,y)
    print 'i:%d, j:%d, r-squared: %f' % (i,j,r_value ** 2)
    plt.scatter(total_angle[i,:], total_angle[j,:])
    plt.show()
    alpha = np.mean(angle_diffs[j])/np.mean(angle_diffs[i])
    #angle_pred = alpha * angle[i,0]
    #if angle_pred > pi:
        #angle_pred[i,0] = angle[i,0] - 2 * pi
#
    #if angle[j,0] > angle[i,0]:
        #bias = angle[j,0] - angle[i,0]
    #else:
        #bias = (angle[j,0]+2*pi) - angle[i,0]
    bias = pi/8
    print alpha, bias
    plt.plot(np.cos(angle[j]), label='data')
    plt.plot(np.cos(alpha*angle[i] + bias), label='model')
    plt.legend()
    plt.show()

Fs, x = get_signal(opt.signal_type)
wc = get_transform(Fs, x, opt.transform_type)

mag, angle = mag_angle(wc)

thresh = np.mean(mag) + 0.1 * np.std(mag)

thresh_angle = np.copy(angle)
thresh_angle[mag < thresh] = math.pi / 3

phase_diff(mag, angle, thresh)

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
subplot(313, "Phase (Thresholded)", thresh_angle) # TODO Overlay
#subplot(313, "Phase", angle)

plt.show(block=False)

analysis(mag, angle, thresh)
