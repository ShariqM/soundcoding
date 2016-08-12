"Speech-Speech Chimera"

from gen_signal import *
from transform import *
from common import *

#FIXME put in direc
opt = options()

Fs, x = gen_signal(opt)
Fs, y = gen_signal(opt)
wavfile.write("test/ss/x.wav", Fs, x.astype(np.int16))
wavfile.write("test/ss/y.wav", Fs, y.astype(np.int16))

for (sig,name) in ((x, "X"), (y, "Y")):
    plot("Signal (%s)" % name, sig)

opt.time_length = len(x)


for f in (1, 6, 18):
    opt.nfilters_per_octave = f
    print 'Using nfilters_per_octave=%d' % f

    wc_x = transform(Fs, x, opt)
    wc_y = transform(Fs, y, opt)

    wc_xy = np.absolute(wc_x) * np.exp(1j * np.angle(wc_y))
    wc_yx = np.absolute(wc_y) * np.exp(1j * np.angle(wc_x))

    names = ['MAG_X_PHASE_Y', 'MAG_Y_PHASE_X']
    t = 0
    for wc_curr in (wc_xy, wc_yx):
        print 'Analyzing %s' % names[t]
        recon = itransform(wc_curr, Fs, opt)
        plot("Reconstruction %d" % t, recon)
        args = (opt.nfilters_per_octave, names[t])
        wavfile.write("test/ss/recon_f=%d_%s.wav" % args, Fs, recon.astype(np.int16))
        t = t + 1

    #plt.show()
