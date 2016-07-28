import numpy as np
import pdb
from numpy import fft
import matplotlib.pyplot as plt
import math
from math import log, ceil, pi, sqrt

def ln(x):
    return log(x, math.e)

def log2(x):
    return log(x, 2)

def mag_angle(wc):
    return np.absolute(wc), np.angle(wc)
