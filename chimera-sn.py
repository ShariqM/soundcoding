"Speech-Noise Chimera"
from gen_signal import *
from transform import *
from common import *

opt = options()

Fs, x = gen_signal(opt)
opt.time_length = len(x)
wc = transform(Fs, x, opt)
wavfile.write("test/orig.wav", Fs, x.astype(np.int16))

plt.figure()
plt.title("Signal")
plt.plot(x)

wc_pc = np.copy(wc) # phase corrupt
wc_pc = np.sign(np.real(wc)) * np.absolute(wc)
wc_pc = np.absolute(wc)

wc_ac = np.copy(wc) # amplitude corrupt
wc_ac = np.mean(np.absolute(wc)) * np.exp(1j * np.angle(wc))

names = ['nothing', 'phase', 'amplitude']
t = 0
for wc_curr in (wc, wc_pc, wc_ac):
    print 'Analyzing %s' % names[t]
    x_recon = itransform(wc_curr, Fs, opt)
    plt.figure()
    plt.title("Reconstruction %d" % t)
    plt.plot(x_recon)
    args = (opt.nfilters_per_octave, t, names[t])
    wavfile.write("test/sn/recon_f=%d_%d_%s_corrupted.wav" % args, Fs, x_recon.astype(np.int16))
    t = t + 1

plt.show()
