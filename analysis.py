import pdb
import sys
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
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
(opt, args) = parser.parse_args()

def get_interesting_filters(mag, thresh):
    filter_points = []
    band_length = 0
    band_thresh = 1

    start = 0
    #fund_k = np.where(mag[:,0] == np.max(mag[:,0]))[0][0]
    #filter_points.append(fund_k)
    #for start in range(fund_k, mag.shape[0]):
        #if mag[start,0] < thresh:
            #break

    for k in range(start, mag.shape[0]):
        if mag[k,0] > thresh:
            band_length = band_length + 1
        if mag[k,0] <= thresh:
            if band_length >= band_thresh:
                section = mag[k-band_length:k,0]
                biggest_k = (k-band_length) + np.where(section == np.max(section))[0][0]
                filter_points.append(biggest_k)
                #filter_points.append(k - band_length/2)
            band_length = 0
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
    i,j = filter_points[0], filter_points[1]

    #i,j = 13, 37
    #i,j = 29, 53
    x,y = total_angle[i,:], total_angle[j,:]
    a, b, r_value, c, std_err = stats.linregress(x,y)
    print 'i:%d, j:%d, r-squared: %f' % (i,j,r_value ** 2)
    plt.scatter(total_angle[i,:], total_angle[j,:])
    plt.show()
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
wc = get_transform(Fs, x)

mag, angle = mag_angle(wc)

thresh = np.mean(mag) + 0.1 * np.std(mag)

thresh_angle = np.copy(angle)
thresh_angle[mag < thresh] = math.pi / 3

def subplot(cmd, title, data, hsv=True):
    ax = plt.subplot(cmd)
    plt.title(title)
    if hsv:
        im = plt.imshow(data, cmap=plt.get_cmap('hsv'))
    else:
        im = plt.imshow(data)
    plt.gca().invert_yaxis()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


plt.figure(1)
plt.subplot(311)
plt.title("Input Signal (%s)" % opt.signal_type)
plt.plot(x)

subplot(312, "Wavegram", mag, hsv=False)
subplot(313, "Phase (Thresholded)", thresh_angle) # TODO Overlay
#subplot(313, "Phase", angle)

plt.show(block=False)

plt.figure(2)
phase_diff(mag, angle, thresh)

plt.figure(3)
analysis(mag, angle, thresh)
