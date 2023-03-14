#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from skimage import morphology


class ViBe:
    '''
    ViBe运动检测，分割背景和前景运动图像
    '''

    def __init__(self, num_sam=21, min_match=2, radiu=20, rand_sam=16):
        self.defaultNbSamples = num_sam  # 每个像素的样本集数量，默认20个
        self.defaultReqMatches = min_match  # 前景像素匹配数量，如果超过此值，则认为是背景像素
        self.defaultRadius = radiu  # 匹配半径，即在该半径内则认为是匹配像素
        self.defaultSubsamplingFactor = rand_sam  # 随机数因子，如果检测为背景，每个像素有1/defaultSubsamplingFactor几率更新样本集和领域样本集

        self.background = 0
        self.foreground = 255

    def __buildNeighborArray(self, img):
        '''
        构建一副图像中每个像素的邻域数组
        参数：输入灰度图像
        返回值：每个像素9邻域数组，保存到self.samples中
        '''
        height, width,channel = img.shape
        self.samples = np.zeros((self.defaultNbSamples, height, width), dtype=np.uint8)

        # 生成随机偏移数组，用于计算随机选择的邻域坐标
        ramoff_xy = np.random.randint(-1, 2, size=(2, self.defaultNbSamples, height, width))

        # xr_=np.zeros((height,width))
        xr_ = np.tile(np.arange(width), (height, 1))
        # yr_=np.zeros((height,width))
        yr_ = np.tile(np.arange(height), (width, 1)).T

        channels = np.zeros((self.defaultNbSamples,1,1))
        xyr_ = np.zeros((2, self.defaultNbSamples, height, width))
        for i in range(self.defaultNbSamples):
            xyr_[1, i] = xr_
            xyr_[0, i] = yr_
            channels[i, 0, :] = np.arange(0, 3).repeat(7)[i]
        xyr_ = xyr_ + ramoff_xy
        # print(channels)
        xyr_[xyr_ < 0] = 0
        tpr_ = xyr_[1, :, :, -1]
        tpr_[tpr_ >= width] = width - 1
        tpb_ = xyr_[0, :, -1, :]
        tpb_[tpb_ >= height] = height - 1
        xyr_[0, :, -1, :] = tpb_
        xyr_[1, :, :, -1] = tpr_

        # xyr=np.transpose(xyr_,(2,3,1,0))
        xyr = xyr_.astype(int)
        channel_ = channels.astype(int)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        B, G, R = cv2.split(img)
        img = np.array([B, G, R])
        # channel=np.random.randint(0,4,size=(self.defaultNbSamples))
        # self.samples = img[xyr[0, :, :, :], xyr[1, :, :, :]]
        self.samples = img[channel_, xyr[0, :, :, :], xyr[1, :, :, :]]

    def ProcessFirstFrame(self, img):
        '''
        处理视频的第一帧
        1、初始化每个像素的样本集矩阵
        2、初始化前景矩阵的mask
        3、初始化前景像素的检测次数矩阵
        参数：
        img: 传入的numpy图像素组，要求灰度图像
        返回值：
        每个像素的样本集numpy数组
        '''
        self.__buildNeighborArray(img)
        self.fgCount = np.zeros((img.shape[0],img.shape[1]))   # 每个像素被检测为前景的次数(计数矩阵)
        self.fgMask = np.zeros((img.shape[0],img.shape[1]))  # 保存前景像素(结果矩阵）

    def Update(self, img):
        '''
        处理每帧视频，更新运动前景，并更新样本集。该函数是本类的主函数
        输入：灰度图像
        '''
        height, width,cha = img.shape
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        # h,s,v=cv2.split(hsv)
        B, G, R = cv2.split(img)
        newImg = np.array([B, G, R])
        # 计算当前像素值与样本库中值之差小于阀值范围RADIUS的个数，采用numpy的广播方法
        # dist = np.abs((self.samples.astype(float) - img.astype(float)).astype(int))
        dstB = np.abs(self.samples.astype(float)[0:7,:,:]-B.astype(float).astype(int))
        dstG = np.abs(self.samples.astype(float)[7:14, :, :] - G.astype(float).astype(int))
        dstR = np.abs(self.samples.astype(float)[14:-1, :, :] - R.astype(float).astype(int))
        dist = np.concatenate((dstB,dstG,dstR))
        # self.defaultRadius =0.13*gray
        # print(self.defaultRadius)
        # dist = dist<self.defaultRadius
        dist[dist < self.defaultRadius] = 1
        dist[dist >= self.defaultRadius] = 0
        matches = np.sum(dist, axis=0)
        # print(dist.shape)
        # print(matches.shape)
        # 如果大于匹配数量阀值，则是背景，matches值False,否则为前景，值True
        matches = matches < self.defaultReqMatches
        self.fgMask[matches] = self.foreground
        self.fgMask[~matches] = self.background
        # 前景像素计数+1,背景像素的计数设置为0
        self.fgCount[matches] = self.fgCount[matches] + 1
        self.fgCount[~matches] = 0
        # 如果某个像素连续50次被检测为前景，则认为一块静止区域被误判为运动，将其更新为背景点
        fakeFG = self.fgCount > 50
        matches[fakeFG] = False
        # 此处是该更新函数的关键
        # 更新背景像素的样本集，分两个步骤
        # 1、每个背景像素有1/self.defaultSubsamplingFactor几率更新自己的样本集
        # 更新样本集方式为随机选取该像素样本集中的一个元素，更新为当前像素的值
        # 2、每个背景像素有1/self.defaultSubsamplingFactor几率更新邻域的样本集
        # 更新邻域样本集方式为随机选取一个邻域点，并在该邻域点的样本集中随机选择一个更新为当前像素值
        # 更新自己样本集
        upfactor = np.random.randint(self.defaultSubsamplingFactor, size=(img.shape[0],img.shape[1]))  # 生成每个像素的更新几率
        upfactor[matches] = 100  # 前景像素设置为100,其实可以是任何非零值，表示前景像素不需要更新样本集
        upSelfSamplesInd = np.where(upfactor == 0)  # 满足更新自己样本集像素的索引
        upSelfSamplesPosition = np.random.randint(self.defaultNbSamples,
                                                  size=upSelfSamplesInd[0].shape)  # 生成随机更新自己样本集的的索引
        samInd = (upSelfSamplesPosition, upSelfSamplesInd[0], upSelfSamplesInd[1])
        upSelfSamplesInd = ((upSelfSamplesPosition/7).astype(int),upSelfSamplesInd[0],upSelfSamplesInd[1])
        self.samples[samInd] = newImg[upSelfSamplesInd]  # 更新自己样本集中的一个样本为本次图像中对应像素值

        # 更新邻域样本集
        upfactor = np.random.randint(self.defaultSubsamplingFactor, size=(img.shape[0],img.shape[1]))  # 生成每个像素的更新几率
        upfactor[matches] = 100  # 前景像素设置为100,其实可以是任何非零值，表示前景像素不需要更新样本集
        upNbSamplesInd = np.where(upfactor == 0)  # 满足更新邻域样本集背景像素的索引
        nbnums = upNbSamplesInd[0].shape[0]
        # print(upNbSamplesInd[0].shape[0])
        # ramNbOffset = np.random.randint(-2, 5, size=(2, nbnums))  # 分别是X和Y坐标的偏移
        # ramNbOffset[1,:]=np.zeros((1, nbnums))  # 分别是X和Y坐标的偏移

        ramoff_y = np.random.randint(-1,2, size=(1, nbnums))
        ramoff_x = np.random.randint(-1, 2, size=(1, nbnums))
        ramNbOffset=np.append(ramoff_x,ramoff_y,axis=0)

        nbXY = np.stack(upNbSamplesInd)
        # print(nbXY.shape)
        nbXY += ramNbOffset
        nbXY[nbXY < 0] = 0
        nbXY[0, nbXY[0, :] >= height] = height - 1
        nbXY[1, nbXY[1, :] >= width] = width - 1
        nbSPos = np.random.randint(self.defaultNbSamples, size=nbnums)
        nbSamInd = (nbSPos, nbXY[0], nbXY[1])
        self.samples[nbSamInd] = newImg[(nbSPos/7).astype(int),upNbSamplesInd[0],upNbSamplesInd[1]]

    def getFGMask(self):
        '''
        返回前景mask
        '''
        return self.fgMask


def main():
    # vc = cv2.VideoCapture("../video/202-233/FormatFactoryPart3.mp4")
    vc = cv2.VideoCapture("../video/103-102/FormatFactoryPart1.mp4")
    # vc = cv2.VideoCapture("../video/blur.mp4")
    c = 0
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vibe = ViBe()

    vibe.ProcessFirstFrame(frame)
    temp=np.zeros(frame.shape,dtype=np.int8)
    d = np.zeros(frame.shape, dtype=np.int8)
    temp1 = frame.copy()
    temp2 = frame.copy()
    while rval:
        rval, frame = vc.read()
        # (b, g, r) = cv2.split(frame)  # 通道分解
        # equ = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        # bH = equ.apply(b)
        # gH = equ.apply(g)
        # rH = equ.apply(r)
        # frame = cv2.merge((bH, gH, rH), )  # 通道合成

        vibe.Update(frame)
        segMat = vibe.getFGMask()
        # 　转为uint8类型
        segMat = segMat.astype(np.uint8)
        segMat = cv2.medianBlur(segMat, 3)#中值滤波
        # segMat = cv2.medianBlur(segMat, 3)  # 中值滤波
        # segMat = cv2.blur(segMat, (6, 6)) #均值滤波


        # 寻找轮廓
        # segMat = morphology.remove_small_objects(segMat, 1860)

        contours, hierarch = cv2.findContours(segMat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area < 150:
                cv2.drawContours(segMat, [contours[i]], 0, 0, -1)

        # d1 = abs(cv2.subtract(segMat, temp1))
        # d2 = abs(cv2.subtract(temp1, temp2))
        # d = cv2.bitwise_and(d1, d2)
        # temp2 = temp1
        # temp1 = segMat

        cv2.imshow("frame",  frame)
        cv2.imshow("SegMat", segMat)
        # cv2.imwrite("./result/" + str(c) + ".jpg", frame,[int(cv2.IMWRITE_PNG_STRATEGY)])
        k = cv2.waitKey(1)
        if k == 27:
            vc.release()
            cv2.destroyAllWindows()
            break
        c = c + 1


if __name__ == '__main__':
    main()
    # vibe=ViBe()
    # vibe.__build