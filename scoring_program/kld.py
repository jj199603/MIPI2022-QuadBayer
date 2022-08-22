import os
import pdb
import numpy as np
import cv2
import matplotlib.pyplot as plt

def read_bin_file(filepath):
    '''
        read '.bin' file

    :param path_bin_file:
        path to '.bin' file

    :return:
        rgbw in numpy array (float32)

    '''
    data = np.fromfile(filepath, dtype=np.uint16)
    ww, hh = data[:2]
    data = data[2:].reshape((hh, ww))
    data = data.astype(np.float32)
    return data

def read_raw_file(filepath, hh, ww):
    '''
        read '.raw' file

    :param path_bin_file:
        path to '.raw' file

    :return:
        rgbw in numpy array (float32)

    '''
    rgbw = np.fromfile(filepath, dtype=np.uint16)
    rgbw = rgbw.reshape((hh, ww))
    rgbw = rgbw.astype(np.float32)
    return rgbw

def get_histogram(data, left_edge, right_edge, n_bins, bin_edges=None):

    '''
        Get normalized histogram from input data

    @param data:
        input data [numpy array]

    @param left_edge:
        x-axis left boundary of histogram [float num]

    @param right_edge:
        x-axis right boundary of histogram [float num]

    @param n_bins:
        number of expected bins [int]

    @param bin_edges:
        defined bin edges [None or 1-D numpy array]

    @return:
        - normalized histogram [1-D numpy array with size (n_bins,)]
        - bin edges used [1-D numpy array with size (n_bins + 1, )]
    '''

    if bin_edges is None:
        data_range = right_edge - left_edge
        bin_width = data_range / n_bins
        bin_edges = np.arange(left_edge, right_edge + bin_width, bin_width)

    #bin_centers = bin_edges[:-1] + (bin_width / 2.0)
    n = np.prod(data.shape)
    hist, _ = np.histogram(data, bin_edges)



    return hist / n, bin_edges



def cal_kld_bayer(bayer_gt, bayer_hat):
    '''
        return symmetric KLD score from 10-bit bayer_gt (target) and bayer_hat (transformed)

    @param bayer_gt:
        input 2D image [numpy array] with range(0, 1023)

    @param bayer_hat:
        input 2D image [numpy array] with range(0, 1023)

    @return:
        symmetric KLD score [float num]
    '''

    # pdb.set_trace()
    # Get normalized histogram
    left_edge, right_edge, n_bins = 0, 1023, 1024
    h_gt, bin_edges = get_histogram(bayer_gt, left_edge, right_edge, n_bins, bin_edges=None)
    h_hat, bin_edges = get_histogram(bayer_hat, left_edge, right_edge, n_bins, bin_edges=None)


    # add one on zero elements
    ww, hh = bayer_gt.shape
    min_val = 1/(ww*hh)
    h_gt = np.where(h_gt != 0, h_gt, min_val)
    h_hat = np.where(h_hat != 0, h_hat, min_val)


    # KL_divergence: D_kl_fwd = sum{h(x) * log[h(x) / h_hat(x)]}
    kl_fwd = np.sum(h_gt  * (np.log(h_gt) - np.log(h_hat)))
    kl_inv = np.sum(h_hat * (np.log(h_hat) - np.log(h_gt)))

    return (kl_fwd + kl_inv)/2



def cal_kld_main(bayer_gt, bayer_out):
    '''
           calculate return symmetric KLD score from 10-bit bayer_gt (target) and bayer_hat (transformed)
           Each channel is calculated separately and its mean value is used.

           kld = (kld_gr + kld_r + kld_g + kld_gb)/4

       @param bayer_gt:
           input 2D image [numpy array] with range(0, 1023)

       @param bayer_hat:
           input 2D image [numpy array] with range(0, 1023)

       @return:
           symmetric KLD score [float num]
       '''
    score_channels = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            score_channels[i, j] = cal_kld_bayer(bayer_gt[i::2, j::2], bayer_out[i::2, j::2])

    # print(score_channels)
    score_mean = np.mean(score_channels)

    return score_mean


