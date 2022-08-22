import os
import sys
import argparse
import time
import random
import numpy as np
import torch, math
import torch.nn as nn
import torch.optim as optim
#from dataset import ImageDatasetRaw
from torch.utils.data import DataLoader
from MIRNet import MIRNet
from collections import OrderedDict
import torch.nn.functional as F
from scoring_program.simple_ISP import isp_util
import cv2
from xml.etree import cElementTree as ET
import struct
import time

def save_bin(filepath, arr):
    arr = np.round(arr).astype('uint16')
    arr = np.clip(arr, 0, 1023)
    height, width = arr.shape
    with open(filepath, 'wb') as fp:
        fp.write(struct.pack('<HH', width, height))
        arr.tofile(fp)

def read_simpleISP_imgIno(path):
    tree = ET.parse(path)
    root = tree.getroot()

    r_gain = root.find('r_gain').text
    b_gain = root.find('b_gain').text
    ccm_00 = root.find('ccm_00').text
    ccm_01 = root.find('ccm_01').text
    ccm_02 = root.find('ccm_02').text
    ccm_10 = root.find('ccm_10').text
    ccm_11 = root.find('ccm_11').text
    ccm_12 = root.find('ccm_12').text
    ccm_20 = root.find('ccm_20').text
    ccm_21 = root.find('ccm_21').text
    ccm_22 = root.find('ccm_22').text
    ccm_matrix = np.array([ccm_00, ccm_01, ccm_02,
                           ccm_10, ccm_11, ccm_12,
                           ccm_20, ccm_21, ccm_22])

    return float(r_gain), float(b_gain), ccm_matrix

def recompose(rggb_bayer):
    '''
        used after denoising
            convert quad bayer pattern to 4-channel rggb tensor
        :param quad_bayer:
            2d tensor [bsz, 4, h/2, w/2]
        :return:
            rggb_bayer: 2d tensor [bsz, 4, h/2, w/2]

    '''
    bsize, h, w = rggb_bayer.shape[0], rggb_bayer.shape[2], rggb_bayer.shape[3]
    result = torch.zeros((bsize, 1, 2*h, 2*w)).cuda()
    result[:, 0, 0:2*h:2, 0:2*w:2] = rggb_bayer[:, 0, :, :]
    result[:, 0, 0:2*h:2, 1:2*w:2] = rggb_bayer[:, 1, :, :]
    result[:, 0, 1:2*h:2, 0:2*w:2] = rggb_bayer[:, 2, :, :]
    result[:, 0, 1:2*h:2, 1:2*w:2] = rggb_bayer[:, 3, :, :]
    #flag1 = result.is_contiguous()
    #flag2 = rggb_bayer.is_contiguous()
    return result

def read_bin_file(filepath):
    '''
    read '.bin' file to 2-d numpy array
    :param path_bin_file:
        path to '.bin' file
    :return:
        2-d image as numpy array (float32)
    '''
    data = np.fromfile(filepath, dtype=np.uint16)
    ww, hh = data[:2]
    data_2d = data[2:].reshape((hh, ww))
    data_2d = data_2d.astype(np.float32)
    return data_2d

def quad_decompose(quad_bayer):
    '''
        used for dataloader
            convert quad bayer pattern to 4-channel rggb tensor
        :param quad_bayer:
            2d tensor [1, h, w]
        :return:
            rggb_bayer: 2d tensor [4, h/2, w/2]

    '''
    h, w = quad_bayer.shape[1], quad_bayer.shape[2]
    # 1-convert quad_bayer to rggb bayer
    col_1 = quad_bayer[:, :, 1::4].contiguous()
    col_2 = quad_bayer[:, :, 2::4].contiguous()
    quad_bayer[:, :, 1::4] = col_2
    quad_bayer[:, :, 2::4] = col_1
    row_1 = quad_bayer[:, 1::4, :].contiguous()
    row_2 = quad_bayer[:, 2::4, :].contiguous()
    quad_bayer[:, 1::4, :] = row_2
    quad_bayer[:, 2::4, :] = row_1
    # 2-decompose rggb bayer to 4-channel rggb tensor
    rggb = torch.cat([ quad_bayer[:, 0:h:2, 0:w:2], quad_bayer[:, 0:h:2, 1:w:2], quad_bayer[:, 1:h:2, 0:w:2], quad_bayer[:, 1:h:2, 1:w:2] ], dim=0)
    return rggb

def padding_and_mask(rggb_bayer):
    target_h = 608
    target_w = 912
    bsize, chan = rggb_bayer.shape[0], rggb_bayer.shape[1]
    tmp = torch.zeros(bsize, chan, target_h, target_w).to(rggb_bayer)
    mask = torch.zeros(bsize, chan, target_h, target_w).to(rggb_bayer)
    tmp[:, :, 0:600, 0:900] = rggb_bayer
    mask[:, :, 0:600, 0:900].fill_(1.0)
    return tmp, mask


INPUT_folder = './Quad_test_dataset_fullres/input/24db/'
#INPUT_folder = './Quad_validation_dataset_fullres/input/24db/'
denoise_net = MIRNet(in_channels=4, out_channels=4).cuda()
device_ids = [Id for Id in range(torch.cuda.device_count())]
denoise_net = nn.DataParallel(denoise_net, device_ids=device_ids)
denoise_net.load_state_dict(torch.load('./checkpoint/mir_24db_aug2.pt'))

start = time.time()

for filename in os.listdir(INPUT_folder):    
    raw_quad_bayer_path =  INPUT_folder + filename
    img_id = raw_quad_bayer_path.split('/')[-1].split('_')[1]
    img_info_path = './Quad_test_dataset_fullres/ImgInfo/quad_' + img_id + '_fullres.xml'
    noise_level = 0.0 / 48.0

    raw_quad_bayer = read_bin_file(raw_quad_bayer_path)
    quad_bayer_normal = raw_quad_bayer.astype(np.float32) / 1023
    quad_bayer_tensor = torch.from_numpy(quad_bayer_normal).float()
    quad_bayer_tensor = torch.unsqueeze(quad_bayer_tensor, 0) # tensor (1, 1200, 1800)
    quad_mulchannel = quad_decompose(quad_bayer_tensor).cuda() # tensor (4, 600, 900)

    H, W = quad_mulchannel.shape[1], quad_mulchannel.shape[2]
    quad_mulchannel = torch.unsqueeze(quad_mulchannel, 0)

    # quad_mulchannel_padding, mask = padding_and_mask(quad_mulchannel)
    # quad_mulchannel_padding = quad_mulchannel_padding.cuda()

    with torch.no_grad():
        denoise_net.eval()
        # denoised_bayer_res = denoise_net(quad_mulchannel_padding)
        denoised_bayer = denoise_net(quad_mulchannel)
        denoised_bayer = torch.clamp(denoised_bayer,0,1) 

    tmp = denoised_bayer * 1023

    input_bayer = recompose(quad_mulchannel)

    denoised_bayer = np.asarray(denoised_bayer.cpu().squeeze() * 1023)
    input_bayer = np.asarray(input_bayer.cpu().squeeze() * 1023)

    save_bin('./output/'+filename, denoised_bayer)

    r_gain, b_gain, CCM = read_simpleISP_imgIno(img_info_path)

    denoised_bayer = np.round(denoised_bayer).astype('uint16')
    denoised_bayer = np.clip(denoised_bayer, 0, 1023)

    denoised_img = isp_util.simple_ISP(denoised_bayer, cfa='RGGB', r_gain=r_gain, b_gain=b_gain, CCM=CCM)
    input_img = isp_util.simple_ISP(input_bayer, cfa='RGGB', r_gain=r_gain, b_gain=b_gain, CCM=CCM)

    save_path_1 = './output/24db/' + img_id + '_24db_denoised.png'
    save_path_2 = './output/24db/' + img_id + '_24db_input.png'
    cv2.imwrite(save_path_1, cv2.cvtColor(denoised_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(save_path_2, cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR))

duration = time.time() - start
print(duration)
