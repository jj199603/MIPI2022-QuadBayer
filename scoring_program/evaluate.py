#!/usr/bin/env python
import cv2
import kld
import simple_ISP.isp_util as isp_util
import pyiqa
import imageio
import subprocess
import shutil
import json
import re
import torch
import numpy as np
from collections import OrderedDict
import os.path
import sys
import os

sys.path.insert(0, os.path.dirname(__file__) + '/simple_ISP/')

input_dir = './sample/input'
output_dir = './sample/output'

submit_dir = os.path.join(input_dir, 'res')
truth_dir = os.path.join(input_dir, 'ref/img')
imgInfo_dir = os.path.join(input_dir, 'ref/imgInfo')


output_filename = os.path.join(output_dir, 'scores.txt')

print(submit_dir)
print(truth_dir)
print(output_dir)
cfa = 'RGGB'


dmsc = isp_util.dmsc_method.demosaic_net
# dmsc = isp_util.dmsc_method.Menon2007


def create_swap_list(list1, list2):
    swap_list = [None] * (len(list1) + len(list2))
    swap_list[::2] = list1
    swap_list[1::2] = list2
    return swap_list


def quad2rggb(quad):
    height, width = quad.shape

    width_axis1 = np.arange(1, width, 4)
    width_axis2 = np.arange(2, width, 4)
    width_swap_quad = create_swap_list(width_axis1, width_axis2)
    width_swap_rggb = create_swap_list(width_axis2, width_axis1)

    height_axis1 = np.arange(1, height, 4)
    height_axis2 = np.arange(2, height, 4)
    height_swap_quad = create_swap_list(height_axis1, height_axis2)
    height_swap_rggb = create_swap_list(height_axis2, height_axis1)

    quad[:, width_swap_rggb] = quad[:, width_swap_quad]
    quad[height_swap_rggb, :] = quad[height_swap_quad, :]

    return quad



if not os.path.isdir(submit_dir):
    print("%s doesn't exist" % submit_dir)

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_res_dir = os.path.join(output_dir, 'res_png')
    output_ref_dir = os.path.join(output_dir, 'ref_png')

    if not os.path.exists(output_res_dir):
        os.makedirs(output_res_dir)

    if not os.path.exists(output_ref_dir):
        os.makedirs(output_ref_dir)


# Creat the evaluation score path
output_filename = os.path.join(output_dir, 'scores.txt')


# Get the path of gt file
img_list = []
# file_list = sorted(os.listdir(submit_dir))
file_list = sorted(os.listdir(truth_dir))


# print('Found %d images in the gt folder.' % len(file_list))

# convert from bayer to rgb

noise_lvl = [0, 24, 42]

for item in file_list:

    # print('item: ', item)

    info_img_name = os.path.join(imgInfo_dir, item.replace('.bin', '.xml'))

    assert os.path.exists(info_img_name), info_img_name + ' does not exist'

    r_gain, b_gain, CCM = isp_util.read_simpleISP_imgIno(info_img_name)

    for i in range(len(noise_lvl)):

        noise = noise_lvl[i]

        res_bayer_name = item.replace('.bin', '_'+str(noise)+'db.bin')
        res_bayer_path = os.path.join(submit_dir, res_bayer_name)

        ref_bayer_path = os.path.join(truth_dir, item)

        assert os.path.exists(
            res_bayer_path), res_bayer_path + ' does not exist'
        assert os.path.exists(
            ref_bayer_path), ref_bayer_path + ' does not exist'

        print('processing ', res_bayer_path)

        res_bayer = kld.read_bin_file(res_bayer_path)

        res_bayer = quad2rggb(res_bayer)

        ref_bayer = kld.read_bin_file(ref_bayer_path)

        res_img = isp_util.simple_ISP(
            res_bayer, cfa, r_gain, b_gain, CCM, dmsc=dmsc)
        ref_img = isp_util.simple_ISP(
            ref_bayer, cfa, r_gain, b_gain, CCM, dmsc=dmsc)

        cv2.imwrite(os.path.join(output_res_dir, res_bayer_name.replace(
            '.bin', '_res.png')), cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_ref_dir, res_bayer_name.replace(
            '.bin', '_ref.png')), cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR))

print('RGB image generated from bayer')


# caluclate IQ metrics


# set up the metris you need.
metrics = ['psnr', 'ssim', 'lpips']
nr_metrics = ['musiq', 'niqe']

score_dict = OrderedDict()
score_dict_all = OrderedDict()

for metric in metrics:
    iqa_metric = pyiqa.create_metric(metric)

    scores = []
    for item in file_list:

        for i in range(len(noise_lvl)):
            noise = noise_lvl[i]

            res_bayer_name = item.replace(
                '.bin', '_' + str(noise) + 'db_res.png')
            ref_bayer_name = item.replace(
                '.bin', '_' + str(noise) + 'db_ref.png')

            res_img = imageio.imread(os.path.join(
                output_res_dir, res_bayer_name))
            ref_img = imageio.imread(os.path.join(
                output_ref_dir, ref_bayer_name))

            assert res_img.dtype == np.uint8 and ref_img.dtype == np.uint8, 'RGB images should be of type uint8'

            res_img = res_img.astype(np.float32)
            ref_img = ref_img.astype(np.float32)

            res_img = torch.tensor(res_img).permute(
                2, 0, 1).unsqueeze_(0) / 255.
            ref_img = torch.tensor(ref_img).permute(
                2, 0, 1).unsqueeze_(0) / 255.

            if metric not in nr_metrics:
                score = iqa_metric(res_img, ref_img).item()
            else:
                score = iqa_metric(res_img).item()  # Non-Reference assessment

            scores.append(score)

    score_dict[metric] = np.mean(scores)
    score_dict_all[metric] = scores


# KL-divergence on the bayer
metric = 'KLD'
scores = []

for item in file_list:
    # idx = item.rfind('_')
    # unique_id = item[0:idx] + '.bin'

    for i in range(len(noise_lvl)):

        noise = noise_lvl[i]

        res_bayer_name = item.replace('.bin', '_'+str(noise)+'db.bin')

        res_bayer = kld.read_bin_file(os.path.join(submit_dir, res_bayer_name))
        ref_bayer = kld.read_bin_file(os.path.join(truth_dir, item))

        # print('KLD: ', res_bayer_name, item )

        score = kld.cal_kld_main(res_bayer, ref_bayer)

        scores.append(score)

score_dict['KLD'] = np.mean(scores)
score_dict_all['KLD'] = scores
score_dict_all['M4'] = np.multiply(np.multiply(np.asarray(score_dict_all['psnr']), np.asarray(score_dict_all['ssim'])),
                                   np.power(2, 1 - np.asarray(score_dict_all['lpips']) - np.asarray(score_dict_all['KLD'])))
score_dict['M4'] = np.mean(score_dict_all['M4'])


# Write the result into score_path/score.txt
with open(output_filename, 'w') as f:
    for metric in metrics:
        f.write('{}: {}\n'.format(metric.upper(), score_dict[metric]))

    f.write('{}: {}\n'.format('KLD', score_dict['KLD']))
    f.write('{}: {}\n'.format('M4', score_dict['M4']))
# shutil.rmtree(output_res_dir)
# shutil.rmtree(output_ref_dir)
