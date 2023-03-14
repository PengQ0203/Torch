import  cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from guidblur import guideFilter
#flatten() 将数组变成一维

if __name__ == '__main__':
    eps = 0.01
    winSize = (3, 3)  # 类似卷积核（数字越大，磨皮效果越好）
    image = cv.imread(r'202-233/10.jpg', cv.IMREAD_ANYCOLOR)
    image = cv.resize(image, None, fx=0.8, fy=0.8, interpolation=cv.INTER_CUBIC)
    I = image / 255.0  # 将图像归一化
    p = I
    s = 3  # 步长
    # guideFilter_img = guideFilter(I, p, winSize, eps, s)

    # 保存导向滤波结果

    # cv.imshow("image", image)
    # cv.imshow("winSize_16", guideFilter_img)
    # img = guideFilter_img.astype("uint16")
    # img=cv.imread("./202-233/2.jpg",0)
    # img = cv.resize(img,(1200,720),interpolation=cv.INTER_CUBIC)
    # img=cv.imread("./blurPic/2.jpg",0)
    # imgcut=img[100:551,500:951]
    # print(imgcut.shape)

    # mean=cv.medianBlur(img,3)

    # hist,bins = np.histogram(mean.flatten(),256,[0,256])
    # cdf = hist.cumsum()
    # cdf_normalized = cdf * hist.max()/ cdf.max()
    # plt.plot(cdf_normalized, color = 'b')
    # plt.hist(img.flatten(),256,[0,256], color = 'r')
    # plt.xlim([0,256])
    # plt.legend(('cdf','histogram'), loc = 'upper left')


    # 绘制一个红色矩形

    ptLeftTop = (380, 100)
    ptRightBottom = (850, 720)
    # 获取原始图像的行和列
    row, col = image.shape[:2]
    # 定义多边形的顶点
    bottom_left = [col * 0.32, row]
    top_left = [col * 0.48, 0]
    top_right = [col * 0.59, 0]
    bottom_right = [col * 0.84, row]
    # 使用顶点定义多边形
    vertices = np.array([bottom_left, top_left, top_right, bottom_right], dtype=np.int32)
    roi_mask = np.zeros((row, col), dtype=np.uint8)
    cv.fillPoly(roi_mask, [vertices], 255)
    # cv.imshow("original", image)
    # cv.imshow("roi_mask", roi_mask)
    roi = cv.bitwise_and(image, image, mask=roi_mask)
    cv.imshow("final_roi.png", roi)
    point_color = (0, 50, 255) # BGR
    thickness = 5
    lineType = 8
    # cv.rectangle(image, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
    # cv.imshow("rec",image)


    # equ = cv.equalizeHist(mean)
    # # res = np.hstack((mean,equ))
    # equ = cv.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    # equ1 = equ.apply(img)
    # plt.figure("img")
    # plt.subplot(1, 2, 1)
    # plt.imshow(img,cmap = plt.cm.gray)
    # plt.subplot(1, 2, 2)
    # # cv.imshow("res",res)
    # plt.imshow(equ1, cmap = plt.cm.gray)
    # plt.figure("plot")
    # hist = cv.calcHist([mean], [0], None, [256], [0, 256])
    # '''
    # # plt.hist(src,pixels)
    # # src:数据源，注意这里只能传入一维数组，使用src.ravel()可以将二维图像拉平为一维数组。
    # # pixels:像素级，一般输入256。
    # # [0, 256] 直方图连在了一起，不输入就是一个个柱子
    # '''
    # plt.subplot(1, 2, 1)
    # plt.hist(mean.ravel(), 256, [0, 256])
    # plt.subplot(1, 2, 2)
    # hist = cv.calcHist([equ], [0], None, [256], [0, 256])
    # plt.hist(equ.ravel(), 256, [0, 256])


    plt.show()
    cv.waitKey()
    cv.destroyAllWindows()
