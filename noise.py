import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
import math
import pdb

N = 2 ** 15

gauss_noise = np.random.randn(N)
F_gnoise = fft.fft(gauss_noise)
Fs = 25000
freqs = fft.fftfreq(N, 1./Fs)

plt.subplot(311)
OneOverF = 1./np.sqrt(np.abs(freqs))
OneOverF[0] = 2 # remove Nan

plt.plot(freqs, OneOverF, label='1/f')
#plt.plot(freqs, np.abs(F_gnoise), label='F')

F_pnoise = F_gnoise * OneOverF
#plt.plot(freqs, np.abs(F_pnoise), label='1/f noise')
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.legend()

plt.subplot(312)
plt.plot(gauss_noise)

plt.subplot(313)
pink_noise = fft.ifft(F_pnoise)
plt.plot(pink_noise)
plt.show()
