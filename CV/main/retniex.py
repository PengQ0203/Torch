# coding=utf-8
import numpy as np
import cv2
import json
from guidblur import guideFilter

def singleScaleRetinex(img, sigma):
    '''单尺度Retinex函数'''

    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
    return retinex


def multiScaleRetinex(img, sigma_list):
    '''多尺度Retinex函数'''
    # 提前分配空间
    retinex = np.zeros_like(img)
    # 遍历所有的尺度
    for sigma in sigma_list:
        # 对计算的结果进行叠加
        retinex += singleScaleRetinex(img, sigma)
    # 计算多个尺度的平均值
    retinex = retinex / len(sigma_list)
    return retinex


def colorRestoration(img, alpha, beta):
    '''颜色灰度函数'''
    img_sum = np.sum(img, axis=2, keepdims=True)
    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))
    return color_restoration


def simplestColorBalance(img, low_clip, high_clip):
    '''最简单的颜色均衡函数'''
    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0
        for u, c in zip(unique, counts):
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c
        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)
    return img


def MSRCR(img, sigma_list, G, b, alpha, beta, low_clip, high_clip):
    '''MSRCR函数'''
    img = np.float64(img) + 1.0
    # 对原图先做多尺度的Retinex
    img_retinex = multiScaleRetinex(img, sigma_list)
    # 对原图做颜色恢复
    img_color = colorRestoration(img, alpha, beta)
    # 进行图像融合
    img_msrcr = G * (img_retinex * img_color + b)

    for i in range(img_msrcr.shape[2]):
        img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                             (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * \
                             255
    # 将图像调整到[0,255]范围内
    img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
    # 做简单的颜色均衡
    img_msrcr = simplestColorBalance(img_msrcr, low_clip, high_clip)
    return img_msrcr


def automatedMSRCR(img, sigma_list):
    '''automatedMSRCR函数'''
    img = np.float64(img) + 1.0
    img_retinex = multiScaleRetinex(img, sigma_list)
    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break

        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255
    img_retinex = np.uint8(img_retinex)
    return img_retinex


def MSRCP(img, sigma_list, low_clip, high_clip):
    '''MSRCP函数'''
    img = np.float64(img) + 1.0
    intensity = np.sum(img, axis=2) / img.shape[2]
    retinex = multiScaleRetinex(intensity, sigma_list)
    intensity = np.expand_dims(intensity, 2)
    retinex = np.expand_dims(retinex, 2)
    intensity1 = simplestColorBalance(retinex, low_clip, high_clip)
    intensity1 = (intensity1 - np.min(intensity1)) / \
                 (np.max(intensity1) - np.min(intensity1)) * \
                 255.0 + 1.0
    img_msrcp = np.zeros_like(img)

    for y in range(img_msrcp.shape[0]):
        for x in range(img_msrcp.shape[1]):
            B = np.max(img[y, x])
            A = np.minimum(256.0 / B, intensity1[y, x, 0] / intensity[y, x, 0])
            img_msrcp[y, x, 0] = A * img[y, x, 0]
            img_msrcp[y, x, 1] = A * img[y, x, 1]
            img_msrcp[y, x, 2] = A * img[y, x, 2]
    img_msrcp = np.uint8(img_msrcp - 1.0)
    return img_msrcp


if __name__ == '__main__':
    # eps = 0.01
    # winSize = (3, 3)  # 类似卷积核（数字越大，磨皮效果越好）
    # image = cv2.imread(r'202-233/10.jpg', cv2.IMREAD_ANYCOLOR)
    # image = cv2.resize(image, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)
    # I = image / 255.0  # 将图像归一化
    # p = I
    # s = 3  # 步长
    # guideFilter_img = guideFilter(I, p, winSize, eps, s)
    #

    # # cv2.imshow("image", image)
    # cv2.imshow("winSize_16", guideFilter_img)

    with open('config.json', 'r') as f:
        config = json.load(f)
        img1_path = '../202-233/10.jpg'
        img = cv2.imread(img1_path)
        # img=img[100:551,500:951]


        img_msrcr = MSRCR(
            img,
            config['sigma_list'],
            config['G'],
            config['b'],
            config['alpha'],
            config['beta'],
            config['low_clip'],
            config['high_clip']
        )

        img_amsrcr = automatedMSRCR(
            img,
            config['sigma_list']
        )

        img_msrcp = MSRCP(
            img,
            config['sigma_list'],
            config['low_clip'],
            config['high_clip']
        )


        cv2.imshow('Image', img)
        cv2.imshow('retinex', img_msrcr)
        cv2.imshow('Automated retinex', img_amsrcr)
        cv2.imshow('MSRCP', img_msrcp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
