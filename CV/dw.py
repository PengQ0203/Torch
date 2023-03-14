import cv2
import numpy as np
from pywt import dwt2, idwt2,wavedec2

# 读取灰度图
img = cv2.imread('./101-102/1.jpg', 0)

# 对img进行haar小波变换：
cA, (cH, cV, cD) = dwt2(img, 'haar')
l1=()
l2=()
l3=()
a,l1,l2,l3=wavedec2(img,3)

# 小波变换之后，低频分量对应的图像：
cv2.imwrite('lena.png', np.uint8(cA / np.max(cA) * 255))
# 小波变换之后，水平方向高频分量对应的图像：
cv2.imwrite('lena_h.png', np.uint8(l3[1] / np.max(cH) * 255))
# 小波变换之后，垂直平方向高频分量对应的图像：
cv2.imwrite('lena_v.png', np.uint8(cV / np.max(cV) * 255))
# 小波变换之后，对角线方向高频分量对应的图像：
cv2.imwrite('lena_d.png', np.uint8(cD / np.max(cD) * 255))

# 根据小波系数重构回去的图像
rimg = idwt2((cA, (cH, cV, cD)), 'haar')
cv2.imwrite('rimg.png', np.uint8(rimg))