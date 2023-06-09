import math
import os
import numpy as np
from numpy import fft
import cv2
import matplotlib.pyplot as plt
import statsmodels.api as smt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 仿真运动模糊
def motion_process(image_size, motion_angle):
    rr=20
    PSF = np.zeros(image_size)
    center_position = (image_size[0] - 1) / 2
    slope_tan = math.tan(motion_angle * math.pi / 180)
    slope_cot = 1 / slope_tan
    if slope_tan <= 1:
        for i in range(rr):
            offset = round(i * slope_tan)  # ((center_position-i)*slope_tan)
            PSF[int(center_position + offset), int(center_position - offset)] = 1
        return PSF / PSF.sum()  # 对点扩散函数进行归一化亮度
    else:
        for i in range(rr):
            offset = round(i * slope_cot)
            PSF[int(center_position - offset), int(center_position + offset)] = 1
        return PSF / PSF.sum()


# 对图片进行运动模糊
def make_blurred(input, PSF, eps):
    input_fft = fft.fft2(input)  # 进行二维数组的傅里叶变换
    PSF_fft = fft.fft2(PSF) + eps
    blurred = fft.ifft2(input_fft * PSF_fft)
    blurred = np.abs(fft.fftshift(blurred))
    return blurred


# 逆滤波
def inverse(input, PSF, eps):
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps  # 噪声功率，这是已知的，考虑epsilon
    result = fft.ifft2(input_fft / PSF_fft)  # 计算F(u,v)的傅里叶反变换
    result = np.abs(fft.fftshift(result))
    return result

# 维纳滤波，K=0.01
def wiener(input, PSF, eps, K=0.01):
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
    result = fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(fft.fftshift(result))
    return result
def selfConv(img):
    p=[]
    for colmn in range(120,200):
        # for row in range(img.shape[0]):
        #     pt=img[row,colmn]*img[row,:]
        temp=img[:,colmn]
        pt=smt.tsa.acf(temp, nlags=450)
        p.append(pt)
    return p

def put(path):
    img = cv2.imread(path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[100:551, 500:951]

    img_h = img.shape[0]
    img_w = img.shape[1]
    # 进行运动模糊处理
    PSF = motion_process((img_h, img_w), 30)
    blurred = np.abs(make_blurred(img, PSF, 1e-3))
    # cv2.imshow("blurred",blurred)
    # print(f.shape[0])
    kernel = np.array([[1.0, 0, -1.0], [2, 0, -2], [1, 0, -1]])
    kernel2 = kernel.transpose()
    dst = cv2.filter2D(blurred, -1, kernel2)
    p = selfConv(dst)
    p = np.asarray(p)
    p = p.transpose()
    print(p.shape)

    x = np.arange(0, 451)

    plt.figure()
    plt.plot(x, p)



    # # 逆滤波
    # res1 = inverse(blurred, PSF, 1e-3)
    # # 维纳滤波
    # res2 = wiener(blurred, PSF, 1e-3)
    #
    # # 添加噪声,standard_normal产生随机的函数
    # blurred_noisy = blurred + 0.1 * blurred.std() * np.random.standard_normal(blurred.shape)
    # # 对添加噪声的图像进行逆滤波
    # res3 = inverse(blurred_noisy, PSF, 0.2 + 1e-3)
    # # 对添加噪声的图像进行维纳滤波
    # res4 = wiener(blurred_noisy, PSF, 0.2 + 1e-3)
    #
    plt.figure()
    plt.subplot(2, 3, 1), plt.axis('off'), plt.imshow(blurred, plt.cm.gray), plt.title('运动模糊')
    # plt.subplot(2, 3, 2), plt.axis('off'), plt.imshow(res1, plt.cm.gray), plt.title(' 逆滤波')
    # plt.subplot(2, 3, 3), plt.axis('off'), plt.imshow(res2, plt.cm.gray), plt.title(' 维纳滤波(k=0.01)')
    # plt.subplot(2, 3, 4), plt.axis('off'), plt.imshow(blurred_noisy, plt.cm.gray), plt.title('有噪声且运动模糊')
    # plt.subplot(2, 3, 5), plt.axis('off'), plt.imshow(res3, plt.cm.gray), plt.title(' 逆滤波')
    # plt.subplot(2, 3, 6), plt.axis('off'), plt.imshow(res4, plt.cm.gray), plt.title(' 维纳滤波(k=0.01)')

    # plt.savefig('3.new.jpg')
    plt.show()


# 图像处理函数，要传入路径
put('../1.png')
