import numpy as np
import os
from isp_util import *
import cv2


def filename2imgInfo(filename, imgInfo_folder):

    name = filename.split('.')[0]
    name = name.split('_')[0] + '_' + name.split('_')[1]

    imgInfo_files = os.listdir(imgInfo_folder)
    out = ''

    for imgInfo_file in imgInfo_files:
        if not imgInfo_file.endswith('.xml'):
            continue

        if imgInfo_file.startswith(name):
            out = imgInfo_file
            break

    return out


if __name__ == '__main__':

    # ------- ISP input / output Director ---------
    img_h, img_w = 1200, 1800 # half res & DN full res
    cfa = 'GBRG'
    INPUT_folder = './sample/input/'   # input dir for Bayer '.bin' files
    imgInfo_folder = './sample/input/' # imgInfo dir for matched Bayer '.xml' files

    OUTPUT_folder = './sample/output/' # output dir

    if os.path.exists(OUTPUT_folder) == False:
        os.mkdir(OUTPUT_folder)

    for filename in os.listdir(INPUT_folder):

        # ------- load Bayer data ---------
        INPUT_path = INPUT_folder + filename

        if not filename.endswith('.bin'):
            continue

        if filename.endswith('.bin'):
            bayer = read_bin_file(INPUT_path)

        # ------- load image info -------------
        print('Start process: ', INPUT_path)

        imgInfo_path = os.path.join(imgInfo_folder, filename2imgInfo(filename, imgInfo_folder))

        if imgInfo_path.endswith('.xml') and os.path.exists(imgInfo_path):

            r_gain, b_gain, CCM = read_simpleISP_imgIno(imgInfo_path)
            print('               using imgInfo file: imgInfo_path')

        else:
            raise Exception('imgInfo of {} not found'.format(filename))

        # ------- Simple ISP ----------------
        bayer_rgb = simple_ISP(bayer, cfa, r_gain, b_gain, CCM)

        # -------- save output image --------------
        save_path = OUTPUT_folder + filename.replace('.bin', '_simpleISP.png')
        cv2.imwrite(save_path, cv2.cvtColor(bayer_rgb, cv2.COLOR_RGB2BGR))

        print('End process: save ISP out (RGB image) done\n----------\n')

