import pdb
import numpy as np
from scipy.io import wavfile
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import librosa
import math

def get_sum(x):
    tot = x[0]
    for sig in x[1:]:
        tot += sig[:len(tot)]
        print 'hi'
    return tot

x = []

plt.subplot(212)
plt.title("Indvidiual")
for i in (440, 446):
    tmp = np.sin(np.linspace(-np.pi, np.pi, i))
    x.append(np.tile(tmp, 100))
    plt.plot(x[-1], label='%d' % i)
plt.legend()

plt.subplot(211)
plt.title("Sum")
plt.plot(get_sum(x))

plt.show()
