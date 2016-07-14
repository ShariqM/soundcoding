import pdb
import numpy as np
from scipy.io import wavfile
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import librosa
import math

x2 = np.sin(np.linspace(-np.pi, np.pi, 201))
x3 = np.sin(np.linspace(-2*np.pi, 2*np.pi, 201))
x4 = np.sin(np.linspace(-3*np.pi, 3*np.pi, 201))
x2 = np.tile(x2, 3)
x3 = np.tile(x3, 3)
x4 = np.tile(x4, 3)
c = x2 + x3 + x4
plt.plot(c)
plt.show()
plt.plot(x2)
plt.plot(x3)
plt.plot(x4)
plt.show()


