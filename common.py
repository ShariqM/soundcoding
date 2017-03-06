import matplotlib.pyplot as plt
from optparse import OptionParser
import numpy as np
from numpy import fft
from math import pi

def options():
    parser = OptionParser()
    parser.add_option("-s", "--signal", dest="signal_type", default="speech",
                      help="(speech, white, pink, harmonic)")

    parser.add_option("-n", "--add_noise", action='store_true', dest="add_noise",
                      default=False)

    parser.add_option("-p", "--subsample_power", type="int", dest="subsample_power",
                      default=0, help="Subsample by 2 ** (arg)")

    parser.add_option("-l", "--length", type="int", dest="time_length",
                      default=8000, help="Number of timepoints to look at")

    parser.add_option("-f", "--filters_per_octave", type="int", dest="nfilters_per_octave",
                      default=12, help="Number of filters per octave")

    parser.add_option("-o", "--number_of_octaves", type="int", dest="noctaves",
                      default=4, help="Number of octaves in transform")

    parser.add_option("-b", "--bandwidth", type="float", dest="bandwidth",
                      default=3, help="Bandwidth of the filter will be 1/bw * octave")

    parser.add_option("-w", "--wavelet_type", dest="wavelet_type",
                      default="sinusoid", help="Type of wavelet (gaussian, sinusoid)")

    parser.add_option("-N", "--transform_length", type="int", dest="N",
                      default=2 ** 20)

    parser.add_option("-T", "--plot_total", action='store_true', dest="plot_total",
                      default=False)

    (opt, args) = parser.parse_args()
    opt.MAX_AMP = 22000

    return opt

def angle_data(angle):
    angle_diff  = np.zeros((angle.shape[0], angle.shape[1] - 1))
    angle_total = np.zeros((angle.shape[0], angle.shape[1] - 1))
    for k in range(angle.shape[0]):
        for i in range(angle.shape[1] - 1):
            a = angle[k,i]
            b = angle[k,i+1]
            if b < a:
                b = b + 2*pi
            angle_diff[k,i] = (b - a)
            if i > 0:
                angle_total[k,i] = angle_total[k,i-1] + (b-a)
            else:
                angle_total[k,i] = a
    return angle_diff, angle_total

def plot_fourier(freqs, Fx, w):
    plt.figure()
    plt.title("Fourier Signal")

    sfreqs = fft.fftshift(freqs)[2**15 - w:2**15 + w+1]
    sFx    = fft.fftshift(Fx)[2**15 - w:2**15 + w+1]

    plt.plot(sfreqs, np.real(sFx), label="Real")
    plt.plot(sfreqs, np.imag(sFx), color='g', label="Imag")
    plt.plot(sfreqs, np.absolute(sFx), color='r', label="Absolute")

    # Handle Axis
    plt.plot([-w, w], [0, 0], color='k')
    top = 1.01 * np.max(np.real(sFx))
    plt.plot([0, 0], [-top, top], color='k')

    plt.legend()
    plt.show()


def plot(title, x, y=None):
    plt.figure()
    plt.title(title)
    if y is not None:
        plt.plot(x,y)
    else:
        plt.plot(x)

def imshow(title, data, subplot=None,hsv=True):
    if subplot is None:
        plt.figure()
        ax = plt.subplot()
    else:
        ax = plt.subplot(subplot)

    plt.title(title, fontsize=18)
    if hsv:
        im = plt.imshow(data, cmap=plt.get_cmap('hsv'), interpolation="nearest")
    else:
        im = plt.imshow(data, interpolation="nearest")
    plt.colorbar()

    ax.set_ylabel("Filter", fontsize=16)
    ax.set_xlabel("Time", fontsize=16)
    plt.gca().invert_yaxis()
    ax.set_aspect('auto') # Fill y-axis
    return ax


