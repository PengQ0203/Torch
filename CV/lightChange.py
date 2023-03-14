import cv2
import numpy as np

# 读取视频
cap = cv2.VideoCapture('../video/202-233/FormatFactoryPart3.mp4')

# 初始化背景模型
bg = cv2.createBackgroundSubtractorMOG2()

# 定义光照补偿函数
def illumination_compensation(img):
    # 将RGB图像转换为HSV空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 获取亮度通道
    v = hsv[:, :, 2]
    # 计算亮度直方图
    hist, bins = np.histogram(v.flatten(), 256, [0, 256])
    # 寻找亮度峰值
    max_peak = np.argmax(hist)
    # 调整亮度值
    if max_peak < 128:
        v[v < max_peak] += max_peak
    else:
        v[v > max_peak] -= max_peak
    # 将调整后的亮度通道与饱和度和色调通道合并为一个新的图像
    hsv[:, :, 2] = v
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        break

    # 去除光照变化
    frame = illumination_compensation(frame)

    # 提取前景
    fgmask = bg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # 显示结果
    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask)

    # 按下q键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
