from collections import OrderedDict
from email.policy import strict
import logging
import os
import time
import torch
import numpy as np
import math
import cv2
import torch.nn as nn


# from tensorboardX import SummaryWriter

def to_tuple_str(str_first, gpu_num, str_ind):
    if gpu_num > 1:
        tmp = '('
        for cpu_ind in range(gpu_num):
            tmp += '(' + str_first + '[' + str(cpu_ind) + ']' + str_ind + ',)'
            if cpu_ind != gpu_num - 1: tmp += ', '
        tmp += ')'
    else:
        tmp = str_first + str_ind
    return tmp


def to_cat_str(str_first, gpu_num, str_ind, dim_):
    if gpu_num > 1:
        tmp = 'torch.cat(('
        for cpu_ind in range(gpu_num):
            tmp += str_first + '[' + str(cpu_ind) + ']' + str_ind
            if cpu_ind != gpu_num - 1: tmp += ', '
        tmp += '), dim=' + str(dim_) + ')'
    else:
        tmp = str_first + str_ind
    return tmp


def to_tuple(list_data, gpu_num, sec_ind):
    out = (list_data[0][sec_ind],)
    for ind in range(1, gpu_num):
        out += (list_data[ind][sec_ind],)
    return out


def log_init(log_dir, name='log'):
    time_cur = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    if os.path.exists(log_dir) == False:
        os.makedirs(log_dir)
    logging.basicConfig(filename=log_dir + '/' + name + '_' + str(time_cur) + '.log',
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def write_tensorboder_logger(logger_path, epoch, **info):
    if os.path.exists(logger_path) == False:
        os.makedirs(logger_path)
    writer = SummaryWriter(logger_path)
    writer.add_scalars('accuracy', {'train_accuracy': info['train_accuracy'], 'test_accuracy': info['test_accuracy']},
                       epoch)
    for tag, value in info.items():
        if tag not in ['train_accuracy', 'test_accuracy']:
            writer.add_scalar(tag, value, epoch)
    writer.close()


def save_arg(args):
    l = len(args.S_ckpt_path.split('/')[-1])
    path = args.S_ckpt_path[:-l]
    if not os.path.exists(path):
        os.makedirs(path)
    f = open(path + 'args.txt', 'w+')
    for key, val in args._get_kwargs():
        f.write(key + ' : ' + str(val) + '\n')
    f.close()


def load_model(model, path):
    logging.info("------------")
    if os.path.exists(path):
        saved_state_dict = torch.load(path)

        model_param = model.state_dict()

        model.load_state_dict(saved_state_dict, strict=False)
        logging.info("load" + str(path))
    else:
        logging.info("=> no model find")
    logging.info("------------")


def print_model_parms(model, string):
    b = []
    for param in model.parameters():
        b.append(param.numel())
    print(sum(b))
    logging.info(string + ': Number of params: %.2fM', sum(b) / 1e6)


def quantize(img, rgb_range):
    pixel_range = 255. / rgb_range
    # return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
    return img.mul(pixel_range).clamp(0, 255).round()


def Tensor2np(tensor_list, rgb_range):
    def _Tensor2numpy(tensor, rgb_range):
        array = np.transpose(quantize(tensor, rgb_range).numpy(), (1, 2, 0)).astype(np.uint8)
        return array

    return [_Tensor2numpy(tensor, rgb_range) for tensor in tensor_list]


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def calc_metrics(img1, img2, crop_border, test_Y=True):
    #
    img1 = img1 / 255.
    img2 = img2 / 255.

    if test_Y and img1.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
        im1_in = rgb2ycbcr(img1)
        im2_in = rgb2ycbcr(img2)
    else:
        im1_in = img1
        im2_in = img2
    height, width = img1.shape[:2]
    if im1_in.ndim == 3:
        cropped_im1 = im1_in[crop_border:height - crop_border, crop_border:width - crop_border, :]
        cropped_im2 = im2_in[crop_border:height - crop_border, crop_border:width - crop_border, :]
    elif im1_in.ndim == 2:
        cropped_im1 = im1_in[crop_border:height - crop_border, crop_border:width - crop_border]
        cropped_im2 = im2_in[crop_border:height - crop_border, crop_border:width - crop_border]
    else:
        raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im1_in.ndim))

    psnr = calc_psnr(cropped_im1 * 255, cropped_im2 * 255)
    ssim = calc_ssim(cropped_im1 * 255, cropped_im2 * 255)
    return psnr, ssim


def calc_psnr(img1, img2):
    # img1 and img2 have range [0, 255]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('80')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')