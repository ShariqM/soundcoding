import pdb
import sys
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
from math import log, ceil, pi, floor
from scipy import stats
from gen_signal import *
from transform import *
from common import *

opt = options()

def get_interesting_filters(mag, thresh):
    candidate_points = []
    band_length = 0
    band_thresh = 1

    start = 0
    mid = mag.shape[1]/2
    for k in range(start, mag.shape[0]):
        if mag[k,mid] > thresh:
            band_length = band_length + 1
        if mag[k,mid] <= thresh:
            if band_length >= band_thresh:
                section = mag[k-band_length:k,mid]
                biggest_k = (k-band_length) + np.where(section == np.max(section))[0][0]
                candidate_points.append(biggest_k)
            band_length = 0

    print (candidate_points)

    top = sorted(zip(mag[candidate_points,mid], candidate_points))[-3:]
    print ('Magnitudes:', mag[candidate_points, mid])
    #filter_points = sorted(zip(*top)[1]) # Broke in python3
    filter_points = sorted([pair[1] for pair in top])
    print ("Filters of interest:", filter_points)
    return filter_points

def analysis(mag, angle, thresh):
    filter_points = get_interesting_filters(mag, thresh)

    plt.figure()
    colors = ["red", "orange", "green", "blue", "purple", "black"]
    #stop = min(len(colors), len(filter_points))
    stop = min(len(filter_points), 4)
    for k in range(1, stop):
        i, j = filter_points[0], filter_points[k]
        plt.plot([-pi, pi], [0, 0], color='k')
        plt.plot([0, 0], [-pi, pi], color='k')
        plt.scatter(angle[i,:], angle[j,:], s=2, color=colors[k-1], label=("%d" % j))
        #plt.scatter(phase[i,:], phase[j,:], s=2, label=("%d" % j))

    plt.title('Phase Correlation (Fund=%d)' % filter_points[0])
    plt.axis('equal')
    plt.legend()
    plt.show()

def phase_diff(mag, angle, thresh):
    angle_diff, angle_total = angle_data(angle)

    filter_points = get_interesting_filters(mag, thresh)

    i,j = filter_points[0], filter_points[1]

    plt.figure()
    plt.plot(angle_total[i,:], angle_total[j,:])
    plt.title("Total Angle")
    plt.xlabel("Filter %d" % i)
    plt.ylabel("Filter %d" % j)

    subsample = 2 ** 2
    x,y = angle_diff[i,:], angle_diff[j,:]
    x,y = x - np.mean(x), y - np.mean(y)
    x   = x[range(0, len(x), subsample)]
    y   = y[range(0, len(y), subsample)]

    plt.figure()
    plt.xlabel("I=%d" % i)
    plt.plot(x)

    a, b, r_value, c, std_err = stats.linregress(x,y)
    print ('i:%d, j:%d, r-squared: %f' % (i,j,r_value ** 2))
    print_every = len(x) / 8

    plt.figure()
    colors = cm.gist_rainbow(np.linspace(0, 1, len(y)))
    for (xi, yi, c, t) in zip(x,y,colors,range(len(y))):
        arg = "t=%d" % (subsample*t) if t % print_every == 0 else None
        plt.scatter(xi, yi, color=c, s=1, label=arg)
    plt.xlabel("Filter %d" % i)
    plt.ylabel("Filter %d" % j)
    plt.legend()

def wavelet_to_spikes(wc):
    mag, angle = mag_angle(wc)
    thresh = np.mean(mag) # - 0.4 * np.std(mag)

    spikes = np.zeros_like(angle)
    for f in range(angle.shape[0]):
        #if f != 13:
            #continue
        for t in range(1, angle.shape[1]):
            if mag[f,t] > thresh and angle[f,t] > 0 and angle[f,t-1] < 0:
            #if angle[f,t] > 0 and angle[f,t-1] < 0:
                spikes[f,t:t+10] = 0.8 if f % 2 == 0 else 1.2
                #print ("Spike, %d" % t)
    return spikes

Fs, x = gen_signal(opt)
wc = transform(Fs, x, opt)

plt.figure()
plt.plot(wc[15,:])

mag, angle = mag_angle(wc)
spikes = wavelet_to_spikes(wc)

thresh = np.mean(mag) + 0.1 * np.std(mag)

thresh_angle = np.copy(angle)
thresh_angle[mag < thresh] = math.pi / 3

plt.figure()
plt.title("Input Signal (%s)" % opt.signal_type)
plt.plot(x)

imshow("Wavegram", mag, hsv=False)

plt.figure()
ax = imshow("Spikes", spikes, subplot=211, hsv=False)
ax = imshow("Wavegram", mag, subplot=212, hsv=False)
#imshow(111, "Phase (Thresholded)", thresh_angle) # TODO Overlay
#imshow("Phase", angle)

#a = np.arange(0, mag.shape[1])
#b = np.arange(0, mag.shape[0])
#A,B = np.meshgrid(a,b)
#contour_thresh = np.mean(mag) + np.std(mag)
#plt.contour(A, B, mag, colors='k', levels=[contour_thresh], linewidths=5)

#phase_diff(mag, angle, contour_thresh)
#analysis(mag, angle, thresh)

plt.show()
