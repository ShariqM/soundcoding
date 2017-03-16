import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from scipy.io import wavfile
from helpers import *
from math import acos, sqrt, pi
from common import *
import pdb


def gaussian(x):
    middle = int(len(x) / 2)
    #250 ** 2
    #var =
    return np.exp(-(x - x[middle]) ** 2 / (N/4))

N = 1000
periods = 10
x = np.linspace(0, 2 *periods * np.pi, N)
r = np.sin(x)
r *= gaussian(x)

Fr = fft.fft(r)




plt.subplot(2,1,1)
plt.plot(r)
plt.plot(gaussian(x))

plt.subplot(2,1,2)
plt.plot(Fr.real)
plt.plot(Fr.imag)
plt.show()
