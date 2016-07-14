import pdb
import numpy as np
from scipy.io import wavfile
import glob
import matplotlib.pyplot as plt
from math import log, ceil, pi

wav_files = glob.glob('data/wavs/s1/*')
Fs, x = wavfile.read(wav_files[0])
plt.plot(x)
plt.show()

hop_length = 128
b = 12 # Filters per octave
Fmin = 50
Fmax = Fmin * 2 ** 7

K = int(ceil(b * log(Fmax/Fmin, 2))) # Number of cq bins
Q = (2 ** (1./b) - 1) ** (-1)

f = np.zeros(K+1)
N = np.zeros(K+1)
timepoints = int(ceil(x.shape[0] / hop_length))
cq = np.zeros((K+1, timepoints), dtype=complex)
print "Fs=%f, Fmin=%f, Fmax=%f, K=%d, Q=%f" % (Fs, Fmin, Fmax, K, Q)
print x.shape[0]

t = 0
for s in range(0, x.shape[0] - 10000, hop_length):
    for k in range(1,K+1):
        f[k] = Fmin * 2 ** (float(k)/b)
        N[k] = ceil(Q * Fs/f[k])
        Nk = int(N[k])
        ns = np.array(range(Nk))
        #print "Nk=%d" % Nk

        e = np.exp(-2 * pi * 1j * (float(Q)/N[k]) * ns)
        h = np.hamming(N[k]) * e
        #print s, Nk, s+Nk
        cq[k,t] = float(1)/Nk * np.dot(x[s:(s+Nk)], h)
        #print cq[k]
        #print "f[%d]=%f, N[%d]=%f" % (k, f[k], k, N[k])
    #print t
    t = t + 1
    #print N

plt.imshow(np.abs(cq))
plt.gca().invert_yaxis()
plt.show()
