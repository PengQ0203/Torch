import numpy as np
import cv2

# cv2.destroyAllWindows()
# cap.release()

def main():
    # vc = cv.VideoCapture("../video/202-233/FormatFactoryPart3.mp4")
    vc = cv2.VideoCapture("../video/103-102/FormatFactoryPart1.mp4")
    # vc = cv.VideoCapture("../video/blur.mp4")
    c = 0

    # ShiTomasi corner detection的参数
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    # 光流法参数
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 创建随机生成的颜色
    color = np.random.randint(0, 255, (100, 3))
    if vc.isOpened():
        ret, old_frame = vc.read()  # 取出视频的第一帧
    else:
        rval = False
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)  # 灰度化
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    mask = np.zeros_like(old_frame)  # 为绘制创建掩码图片

    while rval:
        ret, frame = vc.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 计算光流以获取点的新位置
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # 选择good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # 绘制跟踪框
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)
        k = cv2.waitKey(30)  # & 0xff
        if k == 27:
            break
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        cv2.imshow("frame",frame )
        cv2.imshow("SegMat", dst)
        k = cv2.waitKey(800)
        if k == 25:
            vc.release()
            cv2.destroyAllWindows()
            break
        c = c + 1


if __name__ == '__main__':
    main()