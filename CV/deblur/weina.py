import math
import math

import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


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

# 维纳滤波
def wiener(f, PSF,K=0.1):  # 维纳滤波，K=0.01
    input_fft = np.fft.fft2(f)
    PSF_fft_1 = np.conj(PSF) / (np.abs(PSF) ** 2 + K)
    result = np.fft.ifftshift(np.fft.ifft2(input_fft * PSF_fft_1))
    return result.real


def show(f, s, a, b, c):
    plt.subplot(a, b, c)
    plt.imshow(f, "gray")
    plt.axis('on')
    plt.title(s)


def main():
    img1 = cv2.imread('blur.png', 0)
    img = img1[100:551, 500:951]
    # img = cv.medianBlur(img, 2)
    PSF = get_motion_dsf(img.shape, 4, -0.1)
    plt.figure()
    show(img, "f", 1, 2, 1)
    show(wiener(img, PSF), "restoreImage", 1, 2, 2)
    plt.show()


if __name__ == '__main__':
    main()


