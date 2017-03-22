import pdb
import sys
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
from math import log, ceil, pi, floor
from scipy import stats
from gen_signal import *
from transform import *
from common import *

opt = options()

Fs, x = gen_signal(opt)
print (Fs)
wc = transform(Fs, x, opt)

mag, angle = mag_angle(wc)

#bwig3a.wav
imshow("Wavegram", mag, hsv=False)

def spike(v_bar):
    if v_bar >= threshold:
        return 1
    return 0

def voltage(v_bar):
    if 0 < v_bar < threshold:
        return v_bar
    return 0

N = 16000

plt.figure()
plt.plot(x)

plt.figure()
plt.subplot(3,2,1)
plt.plot(wc[16,:N])
plt.subplot(3,2,2)
plt.plot(wc[27,:N])

ihc = wc[16,:N]
v_bar = np.zeros((2,N))
v = np.zeros((2,N))
a = np.zeros((2,N))
alpha = np.ones((2,N))
tau = 3 * 160
threshold = 30000
for t in range(1,N-2000):
    for i,f in enumerate((16,27)):
        ihc = wc[f,:] + 100 * np.random.rand()
        v_bar[i,t] = alpha[i,t] * ihc[t] + np.exp(-1/tau) * v[i,t-1]
        v[i,t] = voltage(v_bar[i,t])
        a[i,t] = spike(v_bar[i,t])
        if a[i,t] > 0:
            alpha[i,t+1:t+2001] *= np.exp(np.linspace(-1,0,2000))

plt.subplot(3,2,3)
plt.plot(range(N), v[0,:])

plt.subplot(3,2,4)
plt.plot(range(N), v[1,:])

plt.subplot(3,2,5)
height = 1.1 * max(wc[16,:N])
plt.axis([0, N, 0, height])
plt.plot(wc[16,:N])
plt.vlines(np.where(a[0,:] >= 1), 0, 0.5 * height, linewidth=2, color='r')
plt.subplot(3,2,6)
plt.axis([0, N, 0, 2])
plt.vlines(np.where(a[1,:] >= 1), 0, 1)


plt.show()
