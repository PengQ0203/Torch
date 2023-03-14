import cv2
import numpy as np
# 初始化均值漂移参数
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = True
params.blobColor = 255

# 创建Blob检测器
detector = cv2.SimpleBlobDetector_create(params)

# 创建VideoCapture对象
cap = cv2.VideoCapture("../video/103-102/FormatFactoryPart2.mp4")

# 初始化背景图像
ret, bg = cap.read()

# 将背景图像转化为浮点数类型
bg = bg.astype(float)

while True:
    # 读取视频帧
    ret, frame = cap.read()
    # 将当前帧转化为浮点数类型
    # frame = frame.astype(float)
    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 计算灰度直方图并归一化
    hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    # 计算 Gamma 值
    gamma = 0.5
    # 创建 Gamma Lookup Table
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # 进行 Gamma Correction
    gray_gamma = cv2.LUT(gray, table)
    # 显示结果
    cv2.imshow('frame', gray_gamma)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()