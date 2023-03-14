# 利用维纳滤波还原图像
import cv2
import numpy as np
from numpy import fft
from matplotlib import pyplot as plt
import cmath
import math

# 模糊核生成
def get_motion_dsf(image_size, motion_dis, motion_angle):
    PSF = np.zeros(image_size)  # 点扩散函数
    x_center = (image_size[0] - 1) / 2
    y_center = (image_size[1] - 1) / 2

    sin_val = math.sin(motion_angle * math.pi / 180)
    cos_val = math.cos(motion_angle * math.pi / 180)

    # 将对应角度上motion_dis个点置成1
    for i in range(motion_dis):
        x_offset = round(sin_val * i)
        y_offset = round(cos_val * i)
        PSF[int(x_center - x_offset), int(y_center + y_offset)] = 1

    return np.fft.fft2(PSF / PSF.sum())
def degradation_function(m, n,a,b,T):
    P = m / 2 + 1
    Q = n / 2 + 1
    Mo = np.zeros((m, n), dtype=complex)
    for u in range(m):
        for v in range(n):
            temp = cmath.pi * ((u - P) * a + (v - Q) * b)
            if temp == 0:
                Mo[u, v] = T
            else:
                Mo[u, v] = T * cmath.sin(temp) / (temp * cmath.exp(- 1j * temp))
    return Mo


def image_mapping(image):
    img = image/np.max(image)*255
    return img


if __name__ == '__main__':
    img1 = cv2.imread('blur.png', 0)
    img = img1[100:551, 500:951]
    m, n = img.shape
    a = 200
    b = 1e-19
    T = 0.1
    K = 1e-4
    G = fft.fft2(img)
    G_shift = fft.fftshift(G)
    # H = degradation_function(m, n,a,b,T)
    H = get_motion_dsf(img.shape,2,0)
    F = G *((np.abs(H)*np.abs(H)) / (H*np.abs(H)*np.abs(H)+K))

    f_pic = np.abs(fft.ifft2(F))
    res = image_mapping(f_pic)
    res1 = res.astype('uint8')
    res1 = cv2.medianBlur(res1, 3)

    res2 = cv2.equalizeHist(res1)

    plt.subplot(131)
    plt.imshow(img,cmap='gray')
    plt.xlabel('原图', fontproperties='FangSong', fontsize=12)

    plt.subplot(132)
    plt.imshow(res1,cmap='gray')
    plt.xlabel('复原后的图', fontproperties='FangSong', fontsize=12)
    plt.subplot(133)
    plt.imshow(res2,cmap='gray')
    plt.xlabel('复原后的图直方图均衡化', fontproperties='FangSong', fontsize=12)
    plt.show()
