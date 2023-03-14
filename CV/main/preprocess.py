import numpy as np
import cv2
import retniex
import guidblur
import json


def guidBlur(frame, kernalSize, eps, stride):
    winSize = (kernalSize, kernalSize)  # 类似卷积核（数字越大，磨皮效果越好）
    I = frame / 255.0  # 将图像归一化
    p = I
    guideFilter_img = guidblur.guideFilter(I, p, winSize, eps, stride)
    return guideFilter_img


def adjust_gamma(image, gamma=1.0):
    # 对输入的彩色图像进行分离通道
    channels = cv2.split(image)

    # 对每个通道分别进行 Gamma Correction
    corrected_channels = []
    for channel in channels:
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        corrected_channels.append(cv2.LUT(channel, table))

    # 将各个通道合并回去
    corrected_image = cv2.merge(corrected_channels)
    return corrected_image


def process(frame):
    # frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 形态学闭运算+ROI挑选
    # frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, k)
    # frame = cv2.medianBlur(frame, 5)  # 中值滤波
    # frame = cv2.blur(frame, (3, 3))
    frame = ROIregion(frame)
    frame = mean_shift_filter(frame, sp=10, sr=30, max_level=50, eps=1e-6)
    frame = equalHist(frame,1.1)
    # frame = adjust_gamma(frame,1.3)


    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # # 计算灰度直方图并归一化
    # hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
    # cdf = hist.cumsum()
    # cdf_normalized = cdf * hist.max() / cdf.max()
    # # 计算 Gamma 值
    # gamma = 1.2
    # # 创建 Gamma Lookup Table
    # table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # # 进行 Gamma Correction
    # frame = cv2.LUT(gray, table)
    # g1 = cv2.GaussianBlur(frame, (15, 15), 0.2)
    # g2 = cv2.GaussianBlur(frame, (15, 15), 5)
    # frame = cv2.subtract(g1,g2)
    # clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(7, 7))
    # frame = clahe.apply(frame)
    # ret, frame = cv2.threshold(frame, 20, 255, cv2.THRESH_BINARY)
    return frame


# cv2.pow(fi, gamma, frame)
def ROIregion(frame):
    row, col = frame.shape[:2]
    # 202-233的数据

    bottom_left = [col * 0.32, row]
    top_left = [col * 0.48, 0]
    top_right = [col * 0.59, 0]
    bottom_right = [col * 0.84, row]

    # 103-102的数据
    # bottom_left = [col * 0.25, 0.93 * row]
    # bottom_right = [col * 0.8, 0.93 * row]
    # top_left = [col * 0.49, 0.2 * row]
    # top_right = [col * 0.58, 0.2 * row]

    vertices = np.array([bottom_left, top_left, top_right, bottom_right], dtype=np.int32)
    roi_mask = np.zeros((row, col), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [vertices], 255)
    frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
    return frame


def equalHist(image,dd):
    clahe = cv2.createCLAHE(clipLimit=dd, tileGridSize=(7, 7))
    B, G, R = cv2.split(image)  # get single 8-bits channel
    EB = clahe.apply(B)
    EG = clahe.apply(G)
    ER = clahe.apply(R)
    equal_test = cv2.merge((EB, EG, ER))  # merge it back
    return equal_test


def GammaHist(frame, gamma):
    fi = frame / 255.0
    cv2.pow(fi, gamma, frame)
    return frame


def Retniex(img):
    with open('config.json', 'r') as f:
        config = json.load(f)
        # img1_path = '../202-233/10.jpg'
        # img = cv2.imread(img1_path)
        # img=img[100:551,500:951]

        # img = retniex.MSRCR(
        #     img,
        #     config['sigma_list'],
        #     config['G'],
        #     config['b'],
        #     config['alpha'],
        #     config['beta'],
        #     config['low_clip'],
        #     config['high_clip']
        # )

        # img = retniex.automatedMSRCR(
        #     img,
        #     config['sigma_list']
        # )

        img = retniex.MSRCP(
            img,
            config['sigma_list'],
            config['low_clip'],
            config['high_clip']
        )
    return img


def mean_shift_filter(frame, sp=10, sr=30, max_level=50, eps=1e-6):
    row, col = frame.shape[:2]
    # frame1 = img[int(0.2 * row):,int(col * 0.32):int(col * 0.84)]

    scale_percent = 0.1
    # row1, col1 = frame1.shape[:2]
    width = int(row * scale_percent)
    height = int(col * scale_percent)
    dim = (width, height)
    img_ds = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    dst = cv2.pyrMeanShiftFiltering(src=img_ds, sp=15, sr=20)
    result = cv2.resize(dst, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    # img[int(0.2 * row):,int(col * 0.32):int(col * 0.84)] = result
    return result
