import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
def PicProcess(img):
    r=[0,0,0]
    dst=[0,0,0]
    eq=[0,0,0]
    for i in range(len(img)):
        img[i] = img[i][100:401, 500:951]
        equ = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equ = equ.apply(img[i])
        eq[i]=equ
        # eq[i] = cv.medianBlur(eq[i], 9)#中值滤波
        # eq[i] = cv.blur(eq[i], (9, 9)) #均值滤波
        # r[i], dst[i] = cv.threshold(eq[i], 130, 225, 0)  # 202-233
        # r[i], dst[i] = cv.threshold(eq[i], 70, 225, 1)  # blur
        r[i], dst[i] = cv.threshold(eq[i], 90, 225,0)  # 101-102
        dst[i] = cv.medianBlur(dst[i], 9)#中值滤波
        dst[i] = cv.blur(dst[i], (6, 6)) #均值滤波
    return eq,dst
def plotPic(images,equ,dst):
    plt.figure("show")
    for i in range(3):
        plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i], 'gray')
        # plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 2), plt.imshow(equ[i],'gray')
        # plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 3), plt.imshow(dst[i], 'gray')
        # plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
    # plt.show()/
img=[0,0,0]
# # img[0]=cv.imread("./blurPic/1.jpg",0)
# # img[1]=cv.imread("./blurPic/2.jpg",0)
# # img[2]=cv.imread("./blurPic/3.jpg",0)
#
# # img[0]=cv.imread("./202-233/52.jpg",0)
# # img[1] =cv.imread("./202-233/53.jpg",0)
# # img[2] =cv.imread("./202-233/54.jpg",0)
#
# img[0]=cv.imread("./101-102/50.jpg",0)
# img[1] =cv.imread("./101-102/51.jpg",0)
# img[2] =cv.imread("./101-102/52.jpg",0)
# # # current1 = cv.absdiff(img1, img2)  #两帧图象之间的差值，变化的为白色，未变的为黑色
# # # current2 = cv.absdiff(img3, img2)  #两帧图象之间的差值，变化的为白色，未变的为黑色
# # cv.imshow("dd",current1)
# # cv.imshow("dd2",current2)
# # img1 = cv.medianBlur(img1, 5)#中值滤波
# # img2 = cv.medianBlur(img2, 5)#中值滤波
# # img3 = cv.medianBlur(img3, 5)#中值滤波
#
# equ,dst=PicProcess(img)
# plotPic(img,equ,dst)
# # current2 = cv.absdiff(dst[1], dst[0])  #两帧图象之间的差值，变化的为白色，未变的为黑色
# # cv.imshow("current",current2)
#
#
#
# #
# # plt.figure("image3")
# # plt.subplot(1,2,1)
# # plt.imshow(img3, cmap = plt.cm.gray)
# # plt.subplot(1,2,2)
# # plt.imshow(dst3, cmap = plt.cm.gray)
# #
# #
# d1=abs(cv.subtract(dst[1],dst[0]))
# d2=abs(cv.subtract(dst[2],dst[1]))
#
# # plt.figure("D")
# # plt.subplot(1,2,1)
# # plt.imshow(d1, cmap = plt.cm.gray)
# # plt.subplot(1,2,2)
# # plt.imshow(d2, cmap = plt.cm.gray)
# d=cv.bitwise_or(d1,d2)
# # cv.imshow("move",d)

def main():
    # vc = cv.VideoCapture("../video/202-233/FormatFactoryPart3.mp4")
    vc = cv.VideoCapture("../video/103-102/FormatFactoryPart1.mp4")
    # vc = cv.VideoCapture("../video/blur.mp4")
    c = 0
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    d = np.zeros(frame.shape, dtype=np.int8)
    temp1 = frame.copy()
    temp2 = frame.copy()
    while rval:
        rval, frame = vc.read()

        # 将输入转为灰度图
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # equ, dst = PicProcess(gray)
        equ = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equ = equ.apply(gray)
        # eq[i] = cv.medianBlur(eq[i], 9)#中值滤波
        # eq[i] = cv.blur(eq[i], (9, 9)) #均值滤波

        # r, dst = cv.threshold(equ, 130, 225, 0)  # 202-233
        # r, dst = cv.threshold(equ, 70, 225, 1)  # blur
        r, dst = cv.threshold(equ, 100, 225,0)  # 101-102

        dst = cv.medianBlur(dst, 5)#中值滤波
        # dst = cv.blur(dst, (3, 3)) #均值滤波
        dst = cv.GaussianBlur(dst, (5, 5), 0)

        d1 = abs(cv.subtract(dst, temp1))
        d2 = abs(cv.subtract(temp1, temp2))
        d = cv.bitwise_and(d1, d2)
        temp2 = temp1
        temp1 = dst
        cv.imshow("frame",equ )
        cv.imshow("SegMat", dst)
        k = cv.waitKey(800)
        if k == 25:
            vc.release()
            cv.destroyAllWindows()
            break
        c = c + 1


if __name__ == '__main__':
    main()