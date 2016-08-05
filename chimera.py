from optparse import OptionParser
from gen_signal import *
from transform import *

parser = OptionParser()
parser.add_option("-s", "--signal", dest="signal_type", default="speech",
                  help="(speech, white, pink, harmonic)")
parser.add_option("-f", "--filters_per_octave", type="int", dest="nfilters_per_octave",
                  default=12, help="Number of filters per octave")
parser.add_option("-b", "--bandwidth", type="float", dest="bandwidth",
                  default=4, help="Bandwidth of the filter will be 1/bw * octave")
parser.add_option("-p", "--subsample_power", type="int", dest="subsample_power", default=0,
                  help="Subsample by 2 ** (arg)")
parser.add_option("-w", "--wavelet_type", dest="wavelet_type",
                  default="gaussian", help="Type of wavelet (gaussian, sinusoid)")
(opt, args) = parser.parse_args()

Fs, x = gen_signal(opt.signal_type)
opt.time_length = len(x)
wc = transform(Fs, x, opt)
wavfile.write("test/orig.wav", Fs, x.astype(np.int16))

plt.figure()
plt.title("Signal")
plt.plot(x)

wc_pc = np.copy(wc) # phase corrupt
wc_pc = np.absolute(wc)
wc_pc = np.real(wc)

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
    wavfile.write("test/recon_f=%d_%d_%s_corrupted.wav" % args, Fs, x_recon.astype(np.int16))
    t = t + 1

plt.show()
