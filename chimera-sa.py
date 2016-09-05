"Speech-Approximate Chimera"
from gen_signal import *
from transform import *
from common import *
from math import pi
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

opt = options()

Fs, x = gen_signal(opt)
opt.time_length = len(x)
wc = transform(Fs, x, opt)
wavfile.write("test/sa/orig.wav", Fs, x.astype(np.int16))

plot("Signal", x)

mag, angle = mag_angle(wc)
angle_diff, angle_total = angle_data(angle)

imshow("Wavegram", mag[:,:8000], hsv=False)

#plot("Total Angle", angle_total[24,:])

for k in range(wc.shape[0]):
    avg_diff = np.mean(angle_diff[k,:])
    for t in range(1,wc.shape[1]):
        new_angle = angle[k,t-1] + avg_diff
        new_angle = new_angle - 2*pi if new_angle >= pi else new_angle
        angle[k,t] = new_angle

wc_pc = mag
wc_ap = mag * np.exp(1j * angle)
names = ['nothing', 'phase_corrupt', 'phase_replace']
t = 0
for wc_curr in (wc, wc_pc, wc_ap):
    print ('Analyzing %s' % names[t])
    x_recon = itransform(wc_curr, Fs, opt)
    plot("Reconstruction %d" % t, x_recon)
    args = (opt.wavelet_type, opt.nfilters_per_octave, names[t])
    wavfile.write("test/sa/reconstruction_%s_f=%d_%s.wav" % args, Fs, x_recon.astype(np.int16))
    t = t + 1

#plt.show()
