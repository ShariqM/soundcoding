import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0, i1
import pdb

N = 1000
mu, kappa = 0.0, 5.0
s = np.random.vonmises(mu, kappa, N)

total = np.sum(np.cos(s))
print 'Total', total
best_dist = 9999999
best_k = -1.0
for k in np.linspace(0.01, 50, 8000):
    print k, N * (i1(k)/i0(k))
    dist = total - (N * i1(k)/i0(k))
    if np.abs(dist) < best_dist:
        best_dist = dist
        best_k = k
print 'Best K: %f, Dist: %f' % (best_k, best_dist)
pdb.set_trace()

plt.hist(s, 50, normed=True)
x = np.linspace(-np.pi, np.pi, num=51)
y = np.exp(kappa*np.cos(x-mu))/(2*np.pi*i0(kappa))
plt.plot(x, y, linewidth=2, color='r')
plt.show()

x2 = np.linspace(-np.pi, np.pi, num=101)
x2 = x + np.pi/2
