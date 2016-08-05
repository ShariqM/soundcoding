from optparse import OptionParser
import numpy as np
from math import pi

def options():
    parser = OptionParser()
    parser.add_option("-s", "--signal", dest="signal_type", default="speech",
                      help="(speech, white, pink, harmonic)")

    parser.add_option("-p", "--subsample_power", type="int", dest="subsample_power",
                      default=0, help="Subsample by 2 ** (arg)")

    parser.add_option("-l", "--length", type="int", dest="time_length",
                      default=8000, help="Number of timepoints to look at")

    parser.add_option("-f", "--filters_per_octave", type="int", dest="nfilters_per_octave",
                      default=12, help="Number of filters per octave")

    parser.add_option("-b", "--bandwidth", type="float", dest="bandwidth",
                      default=4, help="Bandwidth of the filter will be 1/bw * octave")

    parser.add_option("-w", "--wavelet_type", dest="wavelet_type",
                      default="gaussian", help="Type of wavelet (gaussian, sinusoid)")
    (opt, args) = parser.parse_args()

    return opt

def angle_data(angle):
    angle_diff  = np.zeros((angle.shape[0], angle.shape[1] - 1))
    angle_total = np.zeros((angle.shape[0], angle.shape[1] - 1))
    for k in range(angle.shape[0]):
        for i in range(angle.shape[1] - 1):
            a = angle[k,i]
            b = angle[k,i+1]
            if b < a:
                b = b + 2*pi
            angle_diff[k,i] = (b - a)
            if i > 0:
                angle_total[k,i] = angle_total[k,i-1] + (b-a)
            else:
                angle_total[k,i] = a
    return angle_diff, angle_total


