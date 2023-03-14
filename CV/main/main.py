import cv2
import numpy as np
import param
import torch
import time
import preprocess, postprocess, calloss
import gmm
args = param.Args()
args.set_main_args()

cap = cv2.VideoCapture(args.args.video_path)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
# 设置视频帧频
fps = cap.get(cv2.CAP_PROP_FPS)
# 设置视频大小
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# print(size)
# VideoWritertorch方法是cv2库提供的保存视频方法
# 按照设置的格式来out输出
out = cv2.VideoWriter('./videoOut/103-102out.avi', fourcc, 10, size)

mog = cv2.createBackgroundSubtractorMOG2(varThreshold=25,history=1000)  # 定义高斯混合模型对象 mog
# ret, frame = cap.read()
# row, column, _ = frame.shape
# mog = gmm.GmmModel(frame)

nowframe = 1
ff = 0  # 用以判断间隔多少帧进行算法
img_id, test_list = calloss.loadLabel(args.args.dataset_dir)  # 用已生成测试集的标签和序号
allframeInfo = []  # 用以汇总最后各个图片的指标
q = [0, 0, 0]
alarmFrameset = set()
start = time.time()
while cap.isOpened():  # 摄像头正常，进入循环体，读取摄像头每一帧图像
    ret, frame = cap.read()  # 读取摄像头每一帧图像，frame是这一帧的图像
    frame = preprocess.process(frame)

    if ff % 2 == 0:
        # fgmask = mog.frame_processing(frame)
        fgmask = mog.apply(frame)  # 使用前面定义的高斯混合模型对象 mog 当前帧的运动目标检测，返回二值图像
        gray_frame = fgmask.copy()

        # print(mog.getBackgroundModel())
        gray_frame, count, bboxAll = postprocess.selectBigCoal(gray_frame, frame, 12000)
        xywhAll = torch.tensor(bboxAll)
        real, rightnumber = calloss.calRightNumber(xywhAll, nowframe, frame, img_id, test_list)
        if count != 0:
            q.append(1)
        else:
            q.append(0)
        q.pop(0)
        if sum(q) == args.args.qSize:
            alarmFrameset.add(nowframe)
            alarmFrameset.add(nowframe - 1)
            alarmFrameset.add(nowframe - 2)

        oneframeinfo = [0, 0, 0, 0]
        oneframeinfo[0] = nowframe
        oneframeinfo[1] = count
        oneframeinfo[2] = real
        oneframeinfo[3] = rightnumber
        allframeInfo.append(oneframeinfo)

        calloss.putText(frame, nowframe, count, real, rightnumber)
        cv2.imshow("contours", frame)  # 显示当前帧
        cv2.imshow('output', fgmask)
        # cv2.imshow('output', gray_frame)
        out.write(frame)
        cv2.waitKey(1)

        if (0xff == ord("q")) | (nowframe >= args.args.processFrame):
            break
        ff = ff + 1
    else:
        ff = 0
    nowframe = nowframe + 1
img_id = img_id[img_id <= args.args.processFrame]
alarmFrameset = np.array(list(alarmFrameset), dtype=np.int16)
realNegative = nowframe - len(img_id)
TP = sum(np.isin(alarmFrameset, img_id, invert=False))
FP = len(alarmFrameset) - TP
TN = realNegative - FP
FN = nowframe - len(alarmFrameset) - TN
acc = (TP + TN) / (TP + TN + FP + FN)
pre = TP / (TP + FP)
sen = TP / (TP + FN)
belta = 1
F1 = (1 + belta * belta) * pre * sen / (belta * belta * pre + sen + 1e-10)
f = open("./log/03_14_train.txt", "w")
print("TP true positive is : ", TP, file=f)
print("TN true negative is: ", TN, file=f)
print("FP false positive is: ", FP, file=f)
print("FN false negative is : ", FN, file=f)
print("Accuracy  is : ", acc, file=f)
print("Precision is : ", pre, file=f)  # 越大误报警率越低
print("Recall is : ", sen, file=f)  # 越大漏报警率越低
print("F1  is : ", F1, file=f)

end = time.time()
# allframeInfo = np.array(allframeInfo, dtype=np.int16)
# calloss.printResult("./log/02_24_train.txt", allframeInfo, end - start)
cap.release()  # 释放候选框
out.release()
cv2.destroyAllWindows()
