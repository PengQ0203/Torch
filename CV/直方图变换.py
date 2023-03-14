import cv2
import numpy as np

frame = cv2.imread("./202-233/111.jpg", 1)
row, col = frame.shape[:2]
# 定义多边形的顶点
bottom_left = [col * 0.32, row]
top_left = [col * 0.48, 0]
top_right = [col * 0.59, 0]
bottom_right = [col * 0.84, row]
vertices = np.array([bottom_left, top_left, top_right, bottom_right], dtype=np.int32)
roi_mask = np.zeros((row, col), dtype=np.uint8)
cv2.fillPoly(roi_mask, [vertices], 255)
frame = cv2.medianBlur(frame, 3)  # 中值滤波
frame = cv2.blur(frame, (3, 3))
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 形态学闭运算+ROI挑选
frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, k)
fi = frame / 255.0
# 伽马变换
gamma = 1.3
# out = np.power(fi, gamma)
out = np.power(fi,1.5)
cv2.imshow("img", frame)
cv2.imshow("out", out)
cv2.waitKey()
cv2.destroyAllWindows()