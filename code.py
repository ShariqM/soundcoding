import pdb
import numpy as np
from scipy.io import wavfile
import glob
import matplotlib.pyplot as plt

wav_files = glob.glob('data/wavs/s1/*')
Fs, x = wavfile.read(wav_files[0])
pdb.set_trace()
plt.plot(x)
plt.show()
