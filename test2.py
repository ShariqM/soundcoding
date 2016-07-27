import pdb
import numpy as np
from math import pi, cos, sin, tan

def sec(x):
    y = np.cos(x)
    y[y == 0] = 1e-3
    return 1./y

#a = np.array([-pi, pi/2, 0, pi/2, pi, -pi/2])
#b = np.array([-pi, -3*pi/4, -pi/2, -pi/4, 0, pi/4])

x = np.linspace(-pi, pi - (2*pi/200), 200)
y = 3 * x
#y = np.array([0, pi/2, pi, 3*pi/2, 2*pi, pi/2])
#x = np.array([0, pi/4, pi/2, 3*pi/4, pi, 5*pi/4])

#print x
#print y
#print (b + pi/2) * 2
#print (b * 2) + pi

#bias = np.random.randn() * pi
bias = 0
alpha = 8


eta = 1e-4
for i in range(100000):
    arg = alpha * x + bias
    R = np.sin(y) - np.sin(arg) + np.cos(y) - np.cos(arg)
    if i % 1000 == 0:
        print '%d) a:%f, R: %f' % (i+1, alpha, np.linalg.norm(R))
    #pdb.set_trace()
    trig = np.cos(arg) - np.sin(arg)
    alpha = alpha + eta * np.sum(R * trig * x)
