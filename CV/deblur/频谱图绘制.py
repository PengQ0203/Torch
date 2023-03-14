import cv2 as cv
import numpy as np
from numpy import fft
from matplotlib import pyplot as plt
from sklearn import preprocessing
import cmath
def main():
    # f = plt.imread("1.png")
    # f = cv.cvtColor(f, cv.COLOR_RGB2GRAY)
    f=cv.imread("blur.png", 0)
    f = f[100:551, 500:951]

    # # 生成倒谱图
    # ft = np.fft.fft2(f)
    # ft = np.fft.fftshift(ft)
    # ift = np.abs(np.fft.ifft2(np.fft.ifftshift(np.log(ft))))
    # logF = np.log(ift)

    f = np.fft.fft2(f)
    fshift = np.fft.fftshift(f)
    ft = np.log(np.abs(fshift)+1e-4)
    fft=preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)


    plt.figure("ift")
    plt.imshow(iimg, "gray")
    plt.figure("fft")
    plt.imshow(ft, "gray")
    plt.show()
if __name__ == '__main__':
    main()
