import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
from sklearn.decomposition import FastICA
import glob
import pdb
from optparse import OptionParser
from timeit import default_timer as timer
from math import ceil

np.set_printoptions(threshold=np.nan)

parser = OptionParser()
parser.add_option("-s", "--source", dest="source", default="mix",
                  help="(pitt, environment, mammals, mix)")
(opt, args) = parser.parse_args()

def construct_data(source, N, sz):
    base = 'data/lewicki_audiodata'
    if source == "pitt":
        wav_files = glob.glob('%s/PittSounds/*.wav' % base)
    elif source == "environment":
        wav_files = glob.glob('%s/envsounds/*.wav' % base)
    elif source == "mammals":
        wav_files = glob.glob('%s/mammals/edited/*.wav' % base)
    elif source == "mix":
        wf1 = glob.glob('%s/envsounds/*.wav' % base)
        wf2 = glob.glob('%s/mammals/*.wav' % base)
        ratio = ceil(2*len(wf2)/len(wf1)) # 2 to 1 (env to mammals)
        wav_files = wf1 * ratio + wf2
    else:
        raise Exception("Unknown data source: %s" % source)

    X = np.zeros((N, sz))
    perf = False
    for i in range(N):
        start = timer()
        wfile = np.random.choice(wav_files)
        xFs, x_raw = wavfile.read(wfile)
        #x_raw = resample(x_raw, Fs) # Takes too long for now
        #print ("1", timer() - start) if perf else: pass

        start = np.random.randint(len(x_raw) - sz)
        X[i,:] = x_raw[start:start+sz]
    return X

N = 16000
sz = 128
nfilters = 64
plot_dim = np.ceil(np.sqrt(nfilters))
X = construct_data(opt.source, N, sz)

ica = FastICA(n_components=nfilters, whiten=True)
ica.fit(X)
filters = ica.components_

plotHeight = np.max(np.abs(filters ** 2))
plt.figure()
plt.suptitle("Source: %s" % opt.source, fontsize=24)
for i in range(nfilters):
    axes = plt.subplot(plot_dim, plot_dim, i + 1)
    plt.plot(filters[i,:] ** 2)
    #plt.axis([-plotHeight, plotHeight, 0, sz])
    plt.axis([0, sz, -plotHeight, plotHeight])
    axes.set_xticks([])
    axes.set_yticks([])
print ('Plotting...')
plt.show()

from numpy import fft
plt.figure()
plt.suptitle("Source: %s (Fourier)" % opt.source, fontsize=24)
for i in range(nfilters):
    axes = plt.subplot(plot_dim, plot_dim, i + 1)

    Fx = fft.fft(filters[i,:])
    freqs = fft.fftfreq(sz)
    plt.plot(freqs[sz//2:], np.abs(Fx)[sz//2:])

    #plt.plot(filters[i,:] ** 2)
    #plt.axis([-plotHeight, plotHeight, 0, sz])
    #plt.axis([0, sz, -plotHeight, plotHeight])
    #axes.set_xticks([])
    axes.set_yticks([])
print ('Plotting...')
plt.show()
