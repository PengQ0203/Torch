import cv2
import numpy as np
from matplotlib import pyplot as plt

# 图像
img = cv2.imread('blur.png')
# h参数调节过滤器强度。大的h值可以完美消除噪点，但同时也可以消除图像细节，较小的h值可以保留细节但也可以保留一些噪点
h = 10
# templateWindowSize用于计算权重的模板补丁的像素大小，为奇数，默认7
templateWindowSize = 3
# searchWindowSize窗口的像素大小，用于计算给定像素的加权平均值，为奇数，默认21
searchWindowSize = 21
dst = cv2.fastNlMeansDenoisingColored(img, None, h, h, templateWindowSize, searchWindowSize)
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(dst)
plt.show()

# 视频
cap = cv2.VideoCapture('blur.mp4')
# 创建5个帧的列表
img = [cap.read()[1] for i in range(5)]
# 将所有转化为灰度
gray = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in img]
# 将所有转化为float64
gray = [np.float64(i) for i in gray]
# 创建方差为25的噪声
noise = np.random.randn(*gray[1].shape)*10
# 在图像上添加噪声
noisy = [i for i in gray]
# 转化为unit8
noisy = [np.uint8(np.clip(i, 0, 255)) for i in noisy]
# 对第三帧进行降噪
dst = cv2.fastNlMeansDenoisingMulti(noisy, 2, 5, None, 4, 7, 35)
plt.subplot(131), plt.imshow(gray[2], 'gray')
plt.subplot(132), plt.imshow(noisy[2], 'gray')
plt.subplot(133), plt.imshow(dst, 'gray')
plt.show()


