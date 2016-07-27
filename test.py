import numpy as np
from math import ceil, pi
import pdb

nperiods = 20
period1 = 2 ** 2
period2 = 3 * period1

x1 = np.linspace(-np.pi, np.pi - (2*pi/period1), period1)
x2 = np.linspace(-np.pi, np.pi - (2*pi/period2), period2)
print x1
print x2
pdb.set_trace()
x2 = np.tile(x, nperiods)


N = period
start = np.random.randint(period * nperiods - N)
x1 = x[start:start+N]
start = np.random.randint(period * nperiods - N)
x2 = x[start:start+N]
print x1, x2
pdb.set_trace()
