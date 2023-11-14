# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:09:58 2019

@author: Se Eun Kim, Tae Hee Park, and Il Kyu
Paper: Fast Single Image Dehazing Using Saturation Based Transmission Map Estimation,
To be appear in IEEE Transaction on Image Processing, 2020
"""

import numpy as np
import cv2
from skimage import morphology
import math
from tkinter import filedialog
from tkinter import *
from tkinter.filedialog import askopenfilename
import time
import os


'''
   pixel value: i2f: 0-255 to 0-1, f2i: 0-1 to 0-255 
'''


def i2f(i_image):
    f_image = np.float32(i_image) / 255.0
    return f_image


def f2i(f_image):
    i_image = np.uint8(f_image * 255.0)
    return i_image


'''
    Compute 'A' as described by Tang et al. (CVPR 2014)
'''


def Compute_A_Tang(im):
    # Parameters
    erosion_window = 15
    n_bins = 200

    R = im[:, :, 2]
    G = im[:, :, 1]
    B = im[:, :, 0]

    # compute the dark channel
    dark = morphology.erosion(np.min(im, 2), morphology.square(erosion_window))

    [h, edges] = np.histogram(dark, n_bins)
    numpixel = im.shape[0] * im.shape[1]
    thr_frac = numpixel * 0.99
    csum = np.cumsum(h)
    nz_idx = np.nonzero(csum > thr_frac)[0][0]
    dc_thr = edges[nz_idx]
    mask = dark >= dc_thr
    # similar to DCP till this step
    # next, median of these top 0.1% pixels
    # median of the RGB values of the pixels in the mask
    rs = R[mask]
    gs = G[mask]
    bs = B[mask]

    A = np.zeros((1, 3))

    A[0, 2] = np.median(rs)
    A[0, 1] = np.median(gs)
    A[0, 0] = np.median(bs)

    return A


'''
    Compute intensity: GetIntensity, and Saturation: GetSauration
'''


def GetIntensity(fi):
    return cv2.divide(fi[:, :, 0] + fi[:, :, 1] + fi[:, :, 2], 3)


def GetSaturation(fi, intensity):
    min_rgb = cv2.min(cv2.min(fi[:, :, 0], fi[:, :, 1]), fi[:, :, 2])
    me = np.finfo(np.float32).eps
    S = 1.0 - min_rgb / (intensity + me)
    return S


'''
    Estimate saturation of scene radiance: 3 memthods

'''


def EstimateSaturation(h_saturation, p1):
    p2 = 2.0
    k1 = 0.5 * (1.0 - cv2.pow(1.0 - 2.0 * h_saturation, p1))
    k2 = 0.5 + 0.5 * cv2.pow((h_saturation - 0.5) / 0.5, p2)
    j_saturation = np.where(h_saturation <= 0.5, k1, k2)
    j_saturation = np.maximum(j_saturation, h_saturation)

    return j_saturation


def EstimateSaturation_Quadratic(h_saturation):
    return h_saturation * (2.0 - h_saturation)


def EstimateSaturation_Gamma(h_saturation, g):
    j_saturation = (np.power(h_saturation, 1.0 / g) + 1.0 - np.power(1.0 - h_saturation, 1.0 / g)) / 2.0
    j_saturation = np.maximum(j_saturation, h_saturation)

    return j_saturation


'''
    Estimate Transmission Map
'''


def EstimateTransimission(h_intensity, h_saturation, j_saturation):
    Td = h_intensity * (j_saturation - h_saturation)
    Tmn = j_saturation
    Tmap = 1.0 - (Td / Tmn)
    me = np.finfo(np.float32).eps
    Tmap = np.clip(Tmap, me, 1.0)
    # cv2.imshow('Transmission Map ', f2i(Tmap))
    # cv2.waitKey()
    a = 1
    return Tmap


'''
    Recover dehazed image
'''


def Recover(im, tmap, A):
    res = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / tmap + A[0, ind]
        res[:, :, ind] = np.clip(res[:, :, ind], 0.0, 1.0)

    return res


'''
    Adjust image range
'''


def Adjust(im, perh, perl):
    aim = np.empty(im.shape, im.dtype)
    temp = np.empty(im.shape, im.dtype)
    im_h = np.percentile(im, perh)
    im_l = np.percentile(im, perl)

    for ind in range(0, 3):
        aim[:, :, ind] = (im[:, :, ind] - im_l) / (im_h - im_l)
        temp[:, :, ind] = np.clip(aim[:, :, ind], 0.0, 1.0)
        a = 1
    return aim


'''                
    Normalize image 0 between 1
'''


def Normalize(im):
    aim = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im_h = np.max(im[:, :, ind])
        im_l = np.min(im[:, :, ind])
        aim[:, :, ind] = (im[:, :, ind] - im_l) / (im_h - im_l)
        aim[:, :, ind] = np.clip(aim[:, :, ind], 0.0, 1.0)

    return aim


'''
   White balance using grayworld assumption
'''


def gray_world(im):
    aim = np.empty(im.shape, im.dtype)

    mu_r = np.average(im[:, :, 2])
    mu_g = np.average(im[:, :, 1])
    mu_b = np.average(im[:, :, 0])
    aim[:, :, 0] = np.minimum(im[:, :, 0] * (mu_g / mu_b), 1.0)
    aim[:, :, 2] = np.minimum(im[:, :, 2] * (mu_g / mu_r), 1.0)
    aim[:, :, 1] = im[:, :, 1]

    return aim


'''
  CLAHE
'''


def Clahe(im, clip):
    HSV = cv2.cvtColor(f2i(im), cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    HSV[:, :, 2] = clahe.apply(HSV[:, :, 2])
    result_im = i2f(cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR))

    return result_im

###############################################################################################
def enhance_and_save(input_path, output_path):
    hazy_image = i2f(cv2.imread(input_path, cv2.IMREAD_COLOR))
    A = Compute_A_Tang(hazy_image)
    S_A = np.max(A) - np.min(A)
    hazy_imageWB = gray_world(hazy_image)
    A_WB = Compute_A_Tang(hazy_imageWB)
    S_AWB = np.max(A_WB) - np.min(A_WB)

    perh = 99.9
    perl = 0.5
    epsilon = 0.02
    cl = 1

    if S_A < S_AWB+epsilon:
        hazy_imagen = np.empty(hazy_image.shape, hazy_image.dtype)

        for ind in range(0, 3):
            hazy_imagen[:, :, ind] = hazy_image[:, :, ind] / A[0, ind]

        hazy_imagen = Normalize(hazy_imagen)
        hazy_I = GetIntensity(hazy_imagen)
        hazy_S = GetSaturation(hazy_imagen, hazy_I)

        '''
        Stretch function I: 0.2 (heavy haze) - 0.4 (low haze)
        Stretch function II: no parameter
        Stretch function III: 4.0 (heavy haze) - 2.0 (low haze) 
        '''

        est_S = EstimateSaturation_Gamma(hazy_S, 0.2)
        # est_S = EstimateSaturation_Quadratic(hazy_S)
        # est_S = EstimateSaturation(hazy_S, 2.0)
        Transmap = EstimateTransimission(hazy_I, hazy_S, est_S)
        r_image = Recover(hazy_image, Transmap, A)
        r_image = Adjust(r_image, perh, perl)

    else:
        hazy_imagen = np.empty(hazy_image.shape, hazy_image.dtype)

        for ind in range(0, 3):
            hazy_imagen[:, :, ind] = hazy_image[:, :, ind] / A_WB[0, ind]

        hazy_imagen = Normalize(hazy_imagen)
        hazy_I = GetIntensity(hazy_imagen)
        hazy_S = GetSaturation(hazy_imagen, hazy_I)

        est_S = EstimateSaturation_Gamma(hazy_S, 0.2)
        # est_S = EstimateSaturation_Quadratic(hazy_S)
        # est_S = EstimateSaturation(hazy_S, 2.0)

        Transmap = EstimateTransimission(hazy_I, hazy_S, est_S)
        r_image = Recover(hazy_image, Transmap, A_WB)
        r_image = Adjust(r_image, perh, perl)
        r_image = gray_world(r_image)

    result_ce = Clahe(r_image, cl)
    cv2.imwrite(output_path, f2i(result_ce))


def get_folder_names(parent_folder):
    # Get a list of all items in the parent folder
    items = os.listdir(parent_folder)
    # Filter out only folders
    # folder_names = [parent_folder+'/'+item for item in items]
    return items


def dehaze_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all JPG files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".jpg"):
            input_path = input_folder + '/' + file_name
            output_path = output_folder + '/' + file_name
            # Read the image, apply deHaze, and save the result
            enhance_and_save(input_path, output_path)
###############################################################################################
'''
 Main
'''
## Specify the input and output paths
# input_image_path = "Dataset/UA-DETRAC/hazy/train/MVI_20011_229_0.03/img00141.jpg"
# output_image_path = 'result.jpg'
# # Call the function to enhance and save the image
# enhance_and_save(input_image_path, output_image_path)

train_folder_path = "Dataset/UA-DETRAC/hazy/train"
test_folder_path = "Dataset/UA-DETRAC/hazy/test"
train_output_folder_path = "Dataset/UA-DETRAC/dehaze_ColorCorrection/train"
test_output_folder_path = "Dataset/UA-DETRAC/dehaze_ColorCorrection/test"

train_folder_list = get_folder_names(train_folder_path)
test_folder_list = get_folder_names(test_folder_path)
print(len(train_folder_list), len(test_folder_list))

for folder_name in train_folder_list:
    input_folder = train_folder_path + '/' + folder_name
    output_folder = train_output_folder_path + '/' + folder_name
    print(input_folder, output_folder)
    dehaze_folder(input_folder, output_folder)

for folder_name in test_folder_list:
    input_folder = test_folder_path + '/' + folder_name
    output_folder = test_output_folder_path + '/' + folder_name
    print(input_folder, output_folder)
    dehaze_folder(input_folder, output_folder)

