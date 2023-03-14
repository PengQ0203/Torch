import cv2


def process(gray_frame):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 形态学闭运算
    ret, gray_frame = cv2.threshold(gray_frame, 254, 255, cv2.THRESH_BINARY)
    gray_frame = cv2.morphologyEx(gray_frame, cv2.MORPH_CLOSE, k)
    gray_frame = cv2.medianBlur(gray_frame, 3)  # 中值滤波
    return gray_frame


def selectBigCoal(gray_frame, frame, threhold):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 形态学闭运算
    ret, gray_frame = cv2.threshold(gray_frame, 254, 255, cv2.THRESH_BINARY)
    gray_frame = cv2.morphologyEx(gray_frame, cv2.MORPH_OPEN, k)
    gray_frame = cv2.medianBlur(gray_frame, 3)  # 中值滤波

    gray_frame = process(gray_frame)
    # 使用 findContours 检测图像轮廓框，具体原理有论文，但不建议阅读。会使用即可。
    # 返回值: cnts，轮廓的坐标。 hierarchy，各个框之间父子关系，不常用。
    cnts, hierarchy = cv2.findContours(gray_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制每一个 cnts 框到原始图像 frame 中
    xywhAll = []
    count = 0
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)

        if (w * h) < threhold:  # 根据轮廓c，得到当前最佳矩形框
            # cv2.contourArea(c) < threhold:
            continue
        (x, y, w, h) = cv2.boundingRect(c)  # 根据轮廓c，得到当前最佳矩形框
        xywh = [0, 0, 0, 0]
        xywh[0] = x / gray_frame.shape[1] + (w / gray_frame.shape[1]) / 2
        xywh[1] = y / gray_frame.shape[0] + (h / gray_frame.shape[0]) / 2
        xywh[2] = w / gray_frame.shape[1]
        xywh[3] = h / gray_frame.shape[0]
        xywhAll.append(xywh)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)  # 将该矩形框画在当前帧 frame 上
        count = count + 1
    return gray_frame, count, xywhAll
