from cv2 import dnn_superres
import cv2
import numpy as np

import time#为了计时而已

# 创建SR对象...
sr = dnn_superres.DnnSuperResImpl_create()

# 读图
input = cv2.imread('blur.png', 1)
# 读取模型
sr.readModel("model/EDSR_x4.pb")
# 设定算法和放大比例
sr.setModel("edsr", 4)
# 将图片加载入模型处理，获得超清晰度图片
print("处理图片中...\n");
t0 = time.perf_counter()

upScalePic = sr.upsample(input)
print("处理图片完成\n")
print(time.perf_counter() - t0)

#将图片放大
scale = 4
justBigPic = cv2.resize(input, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

# 输出
cv2.imwrite("pic/justBigPic.jpg", justBigPic)
cv2.imwrite("pic/upScalePic.jpg", upScalePic)
print("输出图片完成\n");
