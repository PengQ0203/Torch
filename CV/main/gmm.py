import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

# 值越大，前景被当作背景的可能性也越大
SIGMA = 6  # 可调整的参数



class GmmModel:
    def __init__(self, sample_image):
        # 像素点个数
        self.img_size = sample_image.shape[0] * sample_image.shape[1]
        # GMM高斯模型的个数 K （这里是固定的，有些方法可以对每个像素进行自适应模型个数K的选取）
        self.k = 5  # 可调整的参数
        # 学习率 Alpha
        self.alpha = 0.5  # 可调整的参数
        # SumOfWeightThreshold T
        self.t = 0.75  # 可调整的参数
        # 各个模型的权重系数（初始化为0）
        self.weight = np.ones(shape=(self.k, sample_image.shape[0], sample_image.shape[1])) / self.k
        # 各高斯模型的均值（初始化为0）
        self.mean = np.zeros(shape=(self.k, sample_image.shape[0], sample_image.shape[1], 3))
        # 各高斯模型的标准差（初始化为默认值）
        self.deviation = np.full(([self.k, sample_image.shape[0], sample_image.shape[1]]), SIGMA)

        np.full([self.k, sample_image.shape[0], sample_image.shape[1]], SIGMA)

    def check(self):
        self.ratio = -1 * (self.weight / self.deviation)
        self.idx = self.ratio.argsort(axis=0)
        self.ratio.sort(axis=0)
        self.ratio *= -1
        cum = np.cumsum(self.ratio, axis=0)
        self.mask_divide = (cum < self.t)
        self.mask_divide = np.choose(self.idx, self.mask_divide)
        # print("shape",self.mask_divide.shape)

    def mahalanobis_probability(self, frame):
        sub = np.subtract(frame, self.mean)
        temp = np.sum(sub ** 2, axis=3) / (self.deviation ** 2)
        self.prob = np.exp(temp / (-2)) / (np.sqrt((2 * np.pi) ** 3) * self.deviation)
        temp = np.sqrt(temp)
        self.mask_distance = (temp <= 2.5 * self.deviation)
        return sub, self.weight, self.deviation

    def update(self, video):
        rho = self.alpha * self.prob

        self.mask_some = np.bitwise_or.reduce(self.mask_distance, axis=0)
        mask_update = np.where(self.mask_some == True, self.mask_distance, -1)

        self.weight = np.where(mask_update == 1, (1 - self.alpha) * self.weight + self.alpha, self.weight)
        self.weight = np.where(mask_update == 0, (1 - self.alpha) * self.weight, self.weight)
        self.weight = np.where(mask_update == -1, 0.0001, self.weight)

        data = np.stack([video] * self.k, axis=0)
        mask = np.stack([mask_update] * 3, axis=3)
        r = np.stack([rho] * 3, axis=3)

        self.mean = np.where(mask == 1, (1 - r) * self.mean + r * data, self.mean)
        self.mean = np.where(mask == -1, data, self.mean)

        self.deviation = np.where(mask_update == 1,
                                  np.sqrt(
                                      (1 - rho) * (self.deviation ** 2) + rho * (
                                          np.sum(np.subtract(video, self.mean) ** 2, axis=3))),
                                  self.deviation)
        self.deviation = np.where(mask_update == -1, 3 + np.ones(shape=(self.k, video.shape[0], video.shape[1])),
                                  self.deviation)

    def result(self, video):
        background = np.zeros(shape=(video.shape[0], video.shape[1], 3), dtype=np.uint8)
        foreground = 255 + np.zeros(shape=(video.shape[0], video.shape[1], 3), dtype=np.uint8)
        m = np.stack([self.mask_some] * 3, axis=2)
        res = np.where(m == False, foreground, background)
        n = np.bitwise_and(self.mask_divide, self.mask_distance)
        n = np.bitwise_or.reduce(n, axis=0)
        n = np.stack([n] * 3, axis=2)
        res = np.where(n == True, background, foreground)
        res = res.sum(axis=2)

        return np.uint8(res)

    def frame_processing(self, video):
        self.check()
        self.mahalanobis_probability(video)
        self.update(video)
        res = self.result(video)
        return res


def main():
    global row, column, K
    cap = cv2.VideoCapture('../../video/202-233/FormatFactoryPart3.mp4')
    ret, frame = cap.read()
    K = 3
    row, column, _ = frame.shape
    gmm = GmmModel(frame)
    while 1:
        ret, frame = cap.read()
        gmm.frame_processing(frame)
        cv2.imshow('frame', frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# main()
