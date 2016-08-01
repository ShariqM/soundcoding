import pdb
import sys
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
from math import log, ceil, pi, floor
from optparse import OptionParser
from scipy import stats
from get_signal import *
from transform import *

parser = OptionParser()
parser.add_option("-s", "--signal", dest="signal_type", default="speech",
                  help="(speech, white, pink, harmonic)")
parser.add_option("-f", "--subsample_power", dest="subsample_power", default=0,
                  help="Subsample by 2 ** (arg)")
parser.add_option("-l", "--length", dest="time_length", default=8000,
                  help="Number of timepoints to look at")
(opt, args) = parser.parse_args()

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

    top = sorted(zip(mag[candidate_points,mid], candidate_points))[-3:]
    print 'Magnitudes:', mag[candidate_points, mid]
    filter_points = sorted(zip(*top)[1])
    print "Filters of interest:", filter_points
    return filter_points

def analysis(mag, angle, thresh):
    filter_points = get_interesting_filters(mag, thresh)

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
    angle_diffs = np.zeros((angle.shape[0], angle.shape[1] - 1))
    total_angle = np.zeros((angle.shape[0], angle.shape[1] - 1))
    for k in range(angle.shape[0]):
        for i in range(angle.shape[1] - 1):
            a = angle[k,i]
            b = angle[k,i+1]
            if b < a:
                b = b + 2*pi
            angle_diffs[k,i] = (b - a)
            if i > 0:
                total_angle[k,i] = total_angle[k,i-1] + (b-a)
            else:
                total_angle[k,i] = a
        #print np.mean(angle_diffs[k]), np.std(angle_diffs[k])

    filter_points = get_interesting_filters(mag, thresh)

    look_corr = True
    if look_corr:
        i,j = filter_points[0], filter_points[1]
        subsample = 2 ** 2

        x,y = angle_diffs[i,:], angle_diffs[j,:]
        x,y = x - np.mean(x), y - np.mean(y)
        x   = x[range(0, len(x), subsample)]
        y   = y[range(0, len(y), subsample)]

        #x,y = total_angle[i,:], total_angle[j,:]


        a, b, r_value, c, std_err = stats.linregress(x,y)
        print 'i:%d, j:%d, r-squared: %f' % (i,j,r_value ** 2)
        print_every = len(x) / 8

        colors = cm.gist_rainbow(np.linspace(0, 1, len(y)))
        for (xi, yi, c, t) in zip(x,y,colors,range(len(y))):
            arg = "t=%d" % (subsample*t) if t % print_every == 0 else None
            plt.scatter(xi, yi, color=c, s=1, label=arg)
        plt.xlabel("Filter %d" % i)
        plt.ylabel("Filter %d" % j)
        plt.legend()

    #for p in filter_points:
        #v = angle_diffs[p,:]
        #plt.plot(v - np.mean(v), label='%d' % p)
    #plt.legend()
    #plt.scatter(angle_diffs[i,:], angle_diffs[j,:])
    return

    alpha = np.mean(angle_diffs[j])/np.mean(angle_diffs[i])
    #angle_pred = alpha * angle[i,0]
    #if angle_pred > pi:
        #angle_pred[i,0] = angle[i,0] - 2 * pi
#
    #if angle[j,0] > angle[i,0]:
        #bias = angle[j,0] - angle[i,0]
    #else:
        #bias = (angle[j,0]+2*pi) - angle[i,0]
    bias = pi/8
    print alpha, bias
    plt.plot(np.cos(angle[j]), label='data')
    plt.plot(np.cos(alpha*angle[i] + bias), label='model')
    plt.legend()
    plt.show()

Fs, x = get_signal(opt.signal_type)
wc = get_transform(Fs, x, opt.time_length, opt.subsample_power)

mag, angle = mag_angle(wc)

thresh = np.mean(mag) + 0.1 * np.std(mag)

thresh_angle = np.copy(angle)
thresh_angle[mag < thresh] = math.pi / 3

def subplot(cmd, title, data, hsv=True):
    ax = plt.subplot(cmd)
    plt.title(title)
    if hsv:
        im = plt.imshow(data, cmap=plt.get_cmap('hsv'), interpolation="nearest")
    else:
        im = plt.imshow(data, interpolation="nearest")
    plt.gca().invert_yaxis()
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.set_aspect('auto') # Fill y-axis
    #plt.colorbar(im, cax=cax)

plt.figure()
plt.subplot(111)
plt.title("Input Signal (%s)" % opt.signal_type)
plt.plot(x)

plt.figure()
subplot(111, "Wavegram", mag, hsv=False)
plt.show()

fig_num = prepare_figure(fig_num)
#subplot(111, "Phase (Thresholded)", thresh_angle) # TODO Overlay
subplot(111, "Phase", angle)

a = np.arange(0, mag.shape[1])
b = np.arange(0, mag.shape[0])
A,B = np.meshgrid(a,b)
contour_thresh = np.mean(mag) + np.std(mag)
plt.contour(A, B, mag, colors='k', levels=[contour_thresh], linewidths=5)

fig_num = prepare_figure(fig_num)
phase_diff(mag, angle, thresh)

#fig_num = prepare_figure(fig_num)
#analysis(mag, angle, thresh)

plt.show()
