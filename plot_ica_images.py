import numpy as np
import matplotlib.pyplot as plt
import pdb

from sklearn.datasets import fetch_mldata
from sklearn.decomposition import FastICA


# fetch natural image patches
image_patches = fetch_mldata("natural scenes data")
X = image_patches.data

# 1000 patches a 32x32
# not so much data, reshape to 16000 patches a 8x8
X = X.reshape(1000, 4, 8, 4, 8)
X = np.rollaxis(X, 3, 2).reshape(-1, 8 * 8)

pdb.set_trace()
# perform ICA
ica = FastICA(n_components=49)
ica.fit(X)
filters = ica.components_

# plot filters
plt.figure()
for i, f in enumerate(filters):
    plt.subplot(7, 7, i + 1)
    plt.imshow(f.reshape(8, 8), cmap="gray")
    plt.axis("off")
plt.show()
