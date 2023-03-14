import cv2

cap = cv2.VideoCapture("../video/202-233/FormatFactoryPart3.mp4")
while cap.isOpened():  # 摄像头正常，进入循环体，读取摄像头每一帧图像
    ret, frame = cap.read()  # 读取摄像头每一帧图像，frame是这一帧的图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 视频处理
    cv2.imshow("origin", frame)  # 显示当前帧
    cv2.imshow('final', gray_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()  # 释放候选框
cv2.destroyAllWindows()
