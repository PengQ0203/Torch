import cv2

cap = cv2.VideoCapture('./videoOut/103-102out.avi')

while (cap.isOpened()):
    # ret返回布尔值
    ret, frame = cap.read()
    # 展示读取到的视频矩阵
    cv2.imshow('image', frame)
    # 键盘等待
    k = cv2.waitKey(0)
    # q键退出
    if k & 0xFF == ord('q'):
        break
# 释放资源
cap.release()
# 关闭窗口
cv2.destroyAllWindows()