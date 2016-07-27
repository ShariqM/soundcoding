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

def get_search_points(mag, thresh):
    points = []
    band_length = 0
    band_thresh = 1

    start = 0
    #fund_k = np.where(mag[:,0] == np.max(mag[:,0]))[0][0]
    #points.append(fund_k)
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
                points.append(biggest_k)
                #points.append(k - band_length/2)
            band_length = 0
    print points
    return points


def analysis(mag, angle, thresh):
    points = get_search_points(mag, thresh)

    plt.figure(2)
    colors = ["red", "orange", "green", "blue", "purple", "black"]
    #stop = min(len(colors), len(points))
    stop = min(len(points), 4)
    for k in range(1, stop):
        i, j = points[0], points[k]
        plt.plot([-pi, pi], [0, 0], color='k')
        plt.plot([0, 0], [-pi, pi], color='k')
        plt.scatter(angle[i,:], angle[j,:], s=2, color=colors[k-1], label=("%d" % j))
        #plt.scatter(phase[i,:], phase[j,:], s=2, label=("%d" % j))

    plt.title('Phase Correlation (Fund=%d)' % points[0])
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

    points = get_search_points(mag, thresh)
    i,j = points[6], points[9]

    #i,j = 13, 37
    #i,j = 29, 53
    x,y = total_angle[i,:], total_angle[j,:]
    a, b, r_value, c, std_err = stats.linregress(x,y)
    print 'i:%d, j:%d, r-squared: %f' % (i,j,r_value ** 2)
    plt.scatter(total_angle[i,:], total_angle[j,:])
    plt.show()
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

phase_diff(mag, angle, thresh)

def subplot(cmd, title, data, hsv=True):
    ax = plt.subplot(cmd)
    plt.title(title)
    if hsv:
        im = plt.imshow(data, cmap=plt.get_cmap('hsv'), interpolation='none')
    else:
        im = plt.imshow(data, interpolation='none')
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

analysis(mag, angle, thresh)
