import numpy as np
import glob
from scipy.io import wavfile
from numpy import fft

def get_signal(name):
    N = 2 ** 15
    Fs = 25000
    top = 8000.

    def scale_up(x):
        biggest = np.max(np.abs(x))
        return x * (top/biggest)

    def get_start(x):
        largest = np.max(np.abs(x))
        for t in range(len(x)):
            if np.abs(x[t]) > 0.3 * largest:
                return max(0, t - 1000)
        raise Exception("Signal not present?")

    def construct_signal(x_raw):
        x = np.zeros(N)
        start = get_start(x_raw)
        fill_length = min(N, len(x_raw[start:]))
        x[:fill_length] = x_raw[start:start+fill_length]
        return x

    if name == "white":
        white_noise = np.floor(scale_up(np.random.randn(N)))
        wavfile.write("test/white.wav", Fs, white_noise.astype(np.int16))
        return Fs, white_noise
    elif name == "pink":
        plot = False
        write = True
        white_noise = np.random.randn(N)
        F_white_noise = fft.fft(white_noise)

        freqs = fft.fftfreq(N, 1./Fs)

        freqs_sub = np.copy(freqs)
        freqs_sub[0] = 0.25
        freq_filter = 1./np.sqrt(np.abs(freqs_sub))

        if plot:
            plt.plot(freqs, freq_filter)
            plt.show()

        F_pink_noise = F_white_noise * freq_filter
        pink_noise = fft.ifft(F_pink_noise)
        pink_noise = np.floor(scale_up(np.real(pink_noise)))
        if plot:
            plt.plot(pink_noise)
            plt.show()
        wavfile.write("test/pink.wav", Fs, pink_noise.astype(np.int16))
        return Fs, pink_noise
    elif name == "speech":
        wav_files = glob.glob('data/wavs/s1_50kHz/*')
        x = np.zeros(N)
        wfile = np.random.choice(wav_files)
        print "Using WAV %s" % wfile
        Fs, x_raw = wavfile.read(wfile)
        x = construct_signal(x_raw)
        wavfile.write("test/sample.wav", Fs, x.astype(np.int16))
        return Fs, x
    elif name == "harmonic":
        period = 2 ** 8
        nperiods = ceil(float(N)/period)
        x = np.zeros(N)
        #offset = np.random.random() * 2 * np.pi + 1
        offset = 0
        #for i in (1,2,4,8,16):
        for i in range(1,5+1):
            #x += np.tile(np.sin(np.linspace(-i*np.pi, i*np.pi, (period+1))), nperiods)[:N]
            #offset = np.random.randint(3) * np.pi/2
            x += np.tile(np.sin(np.linspace(-i*np.pi - offset, i*np.pi - (2*i*pi/period) - offset, period)), nperiods)[:N]
        return Fs, x
    else:
        try:
            Fs, x_raw = wavfile.read('data/mine/%s.wav' % name)
        except Exception as e:
            try:
                Fs, x_raw = wavfile.read('data/wavs/s1_50kHz/%s.wav' % name)
            except Exception as e:
                print "ERROR: Signal file not found"
                sys.exit(0)
        if len(x_raw.shape) > 1:
            x_raw = x_raw[:,0] # remove 2nd channel

        x = construct_signal(x_raw)
        return Fs, x
