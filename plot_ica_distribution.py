import matplotlib.pyplot as plt
import numpy as np
import pdb
from common_ica import construct_data

filters = np.load('dict/mix.npy')


N = 2000
sz = 128
X = construct_data('mix', N, sz)

dim = 3
plot_nfilters = dim ** 2
nfilters = filters.shape[0]
filter_idxs = np.random.choice(range(nfilters), plot_nfilters, replace=False)

for i in range(plot_nfilters):
    print ("Plot %d" % (i+1))
    responses = []
    for j in range(N):
        currentFilter = filters[filter_idxs[i],:]
        responses.append(currentFilter @ X[j,:])
    axes = plt.subplot(dim, dim, i + 1)
    plt.hist(responses, bins='auto')
    axes.set_title("Filter=%d" % filter_idxs[i])
    axes.set_xlabel('Response')
    axes.set_ylabel('Occurences')

plt.show()
