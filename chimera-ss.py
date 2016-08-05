"Speech-Speech Chimera"

from gen_signal import *
from transform import *
from common import *

opt = options()

Fs, x = gen_signal(opt.signal_type)
Fs, y = gen_signal(opt.signal_type)
wavfile.write("test/ss/x.wav", Fs, x.astype(np.int16))
wavfile.write("test/ss/y.wav", Fs, y.astype(np.int16))

opt.time_length = len(x)
wc_x = transform(Fs, x, opt)
wc_y = transform(Fs, y, opt)

for (sig,name) in ((x, "X"), (y, "Y")):
    plt.figure()
    plt.title("Signal (%s)" % name)
    plt.plot(sig)

wc_xy = np.absolute(wc_x) * np.exp(1j * np.angle(wc_y))
wc_yx = np.absolute(wc_y) * np.exp(1j * np.angle(wc_x))

names = ['MAG_X_PHASE_Y', 'MAG_Y_PHASE_X']
t = 0
for wc_curr in (wc_xy, wc_yx):
    print 'Analyzing %s' % names[t]
    recon = itransform(wc_curr, Fs, opt)
    plt.figure()
    plt.title("Reconstruction %d" % t)
    plt.plot(recon)
    args = (opt.nfilters_per_octave, names[t])
    wavfile.write("test/ss/recon_f=%d_%swav" % args, Fs, recon.astype(np.int16))
    t = t + 1

plt.show()
