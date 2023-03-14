# 利用最小二乘法复原图像
import cv2
import numpy as np
from numpy import fft
from matplotlib import pyplot as plt
import cmath,math

def Hcreation(img,l,theta):
    rr = 20
    image_size=img.shape
    PSF = np.zeros(image_size)
    x=l*math.cos(theta)
    y=l*math.sin(theta)
    for u in range(int(x)):
        for v in range(int(y)):
            PSF[u,v]=1/l;
    G = fft.fft2(PSF)
    PSF = fft.fftshift(G)
    return PSF
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
                Mo[u, v] = T * cmath.sin(temp) / temp * cmath.exp(- 1j * temp)
    return Mo

def image_mapping(image):
    img = image/np.max(image)*255
    return img


if __name__ == '__main__':
    img1 = cv2.imread('blur.png', 0)
    img = img1[100:551, 500:951]
    m, n = img.shape
    a = 0.02
    b = 1e-18
    T = 0.01
    r = 0.00008
    G = fft.fft2(img)
    print(G.shape)
    G_shift = fft.fftshift(G)
    print(G_shift.shape)
    FFT=np.abs(np.log(G_shift))
    H = degradation_function(m, n,a,b,T)
    # H=Hcreation(img,2,0)
    p = np.array([[0,-1,0],
                  [-1,4,-1],
                  [0,-1,0]])
    P = fft.fft2(p,[img.shape[0],img.shape[1]])
    F = G_shift*(np.conj(H) / (np.abs(H)**2+r*np.abs(P)**2))

    f_pic = np.abs(fft.ifft2(F))
    res = image_mapping(f_pic)
    res1 = res.astype('uint8')
    #res1 = cv2.medianBlur(res1, 3)
    res2 = cv2.equalizeHist(res1)

    plt.subplot(131)
    plt.imshow(FFT,cmap='gray')
    plt.xlabel('原图', fontproperties='FangSong', fontsize=12)
    plt.subplot(132)
    plt.imshow(res1,cmap='gray')
    plt.xlabel('复原后的图', fontproperties='FangSong', fontsize=12)
    plt.subplot(133)
    plt.imshow(res2,cmap='gray')
    plt.xlabel('复原后的图直方图均衡化', fontproperties='FangSong', fontsize=12)
    plt.show()
