
import numpy as np
import cv2
def getpic(videoPath, svPath):#两个参数，视频源地址和图片保存地址
    cap = cv2.VideoCapture(videoPath)

    numFrame = 0
    while True:
        # 函数cv2.VideoCapture.grab()用来指向下一帧，其语法格式为：
        # 如果该函数成功指向下一帧，则返回值retval为True
        if cap.grab():
            # 函数cv2.VideoCapture.retrieve()用来解码，并返回函数cv2.VideoCapture.grab()捕获的视频帧。该函数的语法格式为：
            # retval, image = cv2.VideoCapture.retrieve()image为返回的视频帧，如果未成功，则返回一个空图像。retval为布尔类型，若未成功，返回False；否则返回True
            flag, frame = cap.retrieve()
            if not flag:
                continue
            else:
                numFrame += 1
                #设置图片存储路径
                newPath = svPath + str(numFrame) + ".jpg"
                # cv2.imencode()函数是将图片格式转换(编码)成流数据，赋值到内存缓存中;主要用于图像数据格式的压缩，方便网络传输。
                cv2.imencode('.jpg', frame)[1].tofile(newPath)
                print(numFrame)
        else:
            break
getpic("../video/202-233/FormatFactoryPart3.mp4", "./202-233/")
print("all is ok")
