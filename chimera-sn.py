"Speech-Noise Chimera"
from gen_signal import *
from transform import *
from common import *

opt = options()
direc = 'test/sn'

Fs, x = gen_signal(opt)
wavfile.write("%s/x.wav" % direc, Fs, x.astype(np.int16))

plot("Signal", x)

opt.time_length = len(x)
for f in (1,6,18):
    opt.nfilters_per_octave = f
    print 'Using nfilters_per_octave=%d' % f

    wc = transform(Fs, x, opt)

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
        plot("Reconstruciton %d" % t, x_recon)
        args = (direc, opt.nfilters_per_octave, t, names[t])
        wavfile.write("%s/recon_f=%d_%d_%s_corrupted.wav" % args, Fs, x_recon.astype(np.int16))
        t = t + 1

    #plt.show()
