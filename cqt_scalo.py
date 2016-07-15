import pdb
import numpy as np
from scipy.io import wavfile
import glob
import matplotlib.pyplot as plt
from math import log, ceil, pi

wav_files = glob.glob('data/wavs/s1/*')
Fs, x = wavfile.read(wav_files[0])
#plt.plot(x)
#plt.show()

b = 1 # Filters per octave
Fmin = 64
Fmax = Fmin * 2 ** 4

K = int(ceil(b * log(Fmax/Fmin, 2))) # Number of cq bins
Q = (2 ** (1./b) - 1) ** (-1)

f = np.zeros(K+1)
N = np.zeros(K+1)
cq = np.zeros((K+1, x.shape[0]), dtype=complex)
length = x.shape[0]
cq = np.zeros((K+1, length), dtype=complex)
print "Fs=%f, Fmin=%f, Fmax=%f, K=%d, Q=%f" % (Fs, Fmin, Fmax, K, Q)

t = 0
for k in range(1,K+1):
    f[k] = Fmin * 2 ** (float(k)/b)
    N[k] = ceil(Q * Fs/f[k])
    Nk = int(N[k])
    ns = np.array(range(Nk))
    e = np.exp(-2 * pi * 1j * (float(Q)/N[k]) * ns)
    h = np.hamming(N[k]) * e
    print "Nk=%d" % Nk
    #for s in range(0, x.shape[0] - Nk, Nk):
    for s in range(0, length - Nk, Nk):
        cq[k,s:s+Nk] = float(1)/Nk * np.dot(x[s:(s+Nk)], h)

plt.imshow(np.abs(cq), aspect='auto', interpolation='none')
plt.gca().invert_yaxis()
plt.show()
