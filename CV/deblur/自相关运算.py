import cv2 as cv
import numpy as np
from numpy import fft
import statsmodels.api as smt
from matplotlib import pyplot as plt
def selfConv(img):
    p=[]
    for colmn in range(150,180):
        # for row in range(img.shape[0]):
        #     pt=img[row,colmn]*img[row,:]
        temp=img[:,colmn]
        pt=smt.tsa.acf(temp, nlags=450)
        p.append(pt)
    return p
def oneConv(img):

    # for row in range(img.shape[0]):
    #     pt=img[row,colmn]*img[row,:]
    temp=img[:,100]
    print(temp.shape)
    pt=smt.tsa.acf(temp, nlags=450)

    # pt=np.correlate(temp,temp, mode='full')

    return pt

# f = plt.imread("1.png")
# f = cv.cvtColor(f, cv.COLOR_RGB2GRAY)
f = cv.imread("blur.png", 0)
f = f[100:551, 500:951]
# print(f.shape[0])

# pone=oneConv(f)
# pone=np.asarray(pone)


kernel = np.array([[1.0,0,-1.0],[2,0,-2],[1,0,-1]])
kernel2=kernel.transpose()
dst = cv.filter2D(f, -1, kernel)
p=selfConv(dst)
p=np.asarray(p)
p=p.transpose()
print(p.shape)
x=np.arange(0,451)

plt.figure()
plt.plot(x,p)
plt.show()
# cv.imshow("dsy",dst)

# f = np.fft.fft2(f)
# fshift = np.fft.fftshift(f)
# ft = np.log(np.abs(fshift) + 1e-4)
#
# ishift = np.fft.ifftshift(fshift)
# iimg = np.fft.ifft2(ishift)
# iimg = np.abs(iimg)
#
# plt.figure("ift")
# plt.imshow(iimg, "gray")
# plt.figure("fft")
# plt.imshow(ft, "gray")
# plt.show()
cv.waitKey()
cv.destroyAllWindows()