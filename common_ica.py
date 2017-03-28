import glob
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
from timeit import default_timer as timer

def construct_data(source, N, sz):
    X = np.zeros((N, sz))

    base = 'data/lewicki_audiodata'
    if source == "pitt":
        wav_files = glob.glob('%s/PittSounds/*.wav' % base)
    elif source == "environment":
        wav_files = glob.glob('%s/envsounds/*.wav' % base)
    elif source == "mammals":
        wav_files = glob.glob('%s/mammals/*.wav' % base)
    elif source == "mix":
        wf1 = glob.glob('%s/envsounds/*.wav' % base)
        wf2 = glob.glob('%s/mammals/*.wav' % base)
        ratio = np.ceil(2*len(wf2)/len(wf1)) # 2 to 1 (env to mammals)
        wav_files = wf1 * ratio + wf2
    elif source == "white":
        for i in range(N):
            X[i,:] = np.random.randn(sz)
        return X
    else:
        raise Exception("Unknown data source: %s" % source)

    perf = False
    for i in range(N):
        start = timer()
        wfile = np.random.choice(wav_files)
        xFs, x_raw = wavfile.read(wfile)
        #x_raw = resample(x_raw, Fs) # Takes too long for now
        #print ("1", timer() - start) if perf else: pass

        start = np.random.randint(len(x_raw) - sz)
        X[i,:] = x_raw[start:start+sz]
    return X


