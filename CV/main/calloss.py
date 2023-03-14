import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import cv2


def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter)


def process_iou(detections, line, lablesList, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    label = lablesList[line, :].reshape(3, 5)
    labels = label[[not np.all(label[i] == 0) for i in range(label.shape[0])], :]

    labels = torch.tensor(labels[:, 1:])
    labelsxy = xywh2xyxy(labels)
    # correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    # print("labels is :",labelsxy[:, :])
    # print("detection is ",detections[:, :4])
    iou = box_iou(labelsxy[:, :], detections[:, :4])
    correct = int(((iou > iouv[0]) == True).sum())
    # print(iou)
    return iou, correct


def loadLabel(dataset_dir):
    csv_dir = os.path.join(dataset_dir, "train")
    # dataset_dir = "../../bigcoal/part3/train"
    mode = "train"
    # img_list_txt = os.path.join(dataset_dir, mode + ".txt")  # 储存图片位置的列表
    label_csv = os.path.join(csv_dir, mode + ".csv")  # 储存标签的数组文件
    label = np.loadtxt(label_csv).astype(np.float)  # 读取标签数组文件
    image_list = os.listdir(os.path.join(dataset_dir, "labels"))
    image_id = np.array([int(x.split('.')[0]) for x in image_list]).reshape(-1, 1)
    test_csv = np.concatenate((label, image_id), axis=1)

    return image_id, label


def isInList(imgid, imgID_list):
    if imgid in imgID_list:
        res = np.where(imgID_list[:, -1] == imgid)
        line = res[0][0]
        print(imgID_list[line])
        return line
    else:
        return -1


def xywh2xyxy(x):
    # print(x.shape)
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def printResult(path, allframeInfo,time):
    ''' allFrameInfo: nowframe,count,real,rightnumber'''

    fps = len(allframeInfo) /time
    negative = allframeInfo[allframeInfo[:, 2] == 0]  # 实际负样本
    TP = (allframeInfo[:, 3] != 0).sum()
    TN = (negative[:, 1] == 0).sum()
    FP = len(negative) - TN
    FN = (allframeInfo[:, 2] != 0).sum() - TP
    acc = (TP + TN) / (TP + TN + FP + FN)
    pre = TP / (TP + FP)
    sen = TP / (TP + FN)
    belta = 1
    F1 = (1 + belta * belta) * pre * sen / (belta * belta * pre + sen + 1e-10)
    f = open(path, "w")
    # np.savetxt('./log/NoFlurframe.txt',allframeInfo)
    print("TP true positive is : ", TP, file=f)
    print("TN true negative is: ", TN, file=f)
    print("FP false positive is: ", FP, file=f)
    print("FN false negative is : ", FN, file=f)
    print("Accuracy  is : ", acc, file=f)
    print("Precision is : ", pre, file=f)  # 越大误报警率越低
    print("Recall is : ", sen, file=f)  # 越大漏报警率越低
    print("F1  is : ", F1, file=f)
    print("FPS is : ", fps, file=f)


def putText(frame, nowframe, count, real, rightnumber):
    text = 'Frame is : '
    fontFace = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 1
    fontcolor = (255, 255, 0)  # BGR
    thickness = 2
    lineType = 4
    cv2.putText(frame, text + str(nowframe), (1000, 200), fontFace, fontScale, fontcolor, thickness, lineType)
    # cv2.putText(frame, "predict " + str(count) + " big coal", (1000, 250), fontFace, fontScale, fontcolor, thickness,
    #             lineType)
    # cv2.putText(frame, "real " + str(real) + " big coal", (1000, 300), fontFace, fontScale, fontcolor, thickness,
    #             lineType)
    # cv2.putText(frame, "right prediction is : " + str(rightnumber), (1000, 350), fontFace, fontScale, fontcolor,
    #             thickness, lineType)
    # cv2.putText(frame, "wrong prediction is : " + str(count - rightnumber), (1000, 400), fontFace, fontScale,
    #             (0, 0, 255), thickness, lineType)
    # cv2.putText(frame, "mised prediction is : " + str(real - rightnumber), (1000, 450), fontFace, fontScale, fontcolor,
    #             thickness, lineType)


def calRightNumber(xywhAll, nowframe, frame, img_id, test_list):
    if len(xywhAll) != 0:
        bboxxyxy = xywh2xyxy(xywhAll)
    else:
        bboxxyxy = torch.tensor([[0, 0, 0, 0]])
    flag = isInList(nowframe, img_id)
    if flag == -1:
        err = 0
        rightnumber = 0
        real = 0
    else:
        err, correct = process_iou(bboxxyxy, flag, test_list, torch.tensor([0.08]))
        label = test_list[flag, :].reshape(3, 5)
        labels = label[[not np.all(label[i] == 0) for i in range(label.shape[0])], :]
        real = len(labels)
        for i in range(len(labels)):
            cv2.rectangle(frame, (int(labels[i, 1] * frame.shape[1] - labels[i, 3] * frame.shape[1] / 2), \
                                  int(labels[i, 2] * frame.shape[0] - labels[i, 4] * frame.shape[0] / 2)), \
                          (int(labels[i, 1] * frame.shape[1] + labels[i, 3] * frame.shape[1] / 2), \
                           int(labels[i, 2] * frame.shape[0] + labels[i, 4] * frame.shape[0] / 2)), \
                          (55, 55, 255), 2)  # 将该矩形框画在当前帧 frame 上
        rightnumber = correct
    return real, rightnumber


if __name__ == '__main__':
    labels = torch.tensor([[0.3875, 0.2507, 0.5090, 0.1657]])
    detections = torch.tensor([[0.3347, 0.9343, 0.3771, 1.0000],
                               [0.4222, 0.7259, 0.4604, 0.8037],
                               [0.3764, 0.5046, 0.4069, 0.6204],
                               [0.2590, 0.3333, 0.3521, 0.6287],
                               [0.4722, 0.2028, 0.5340, 0.2796]])
    dataset_dir = "../../bigcoal/part3"
    img_id, test_list = loadLabel(dataset_dir)
    flag = isInList(33, img_id)
    print(flag)
    if flag == -1:
        err = 0
    else:
        err = process_iou(detections, flag, test_list, torch.tensor([0.2]))
    print(err)
    a = np.array([[0.2, 0.2, 0.5, 0.5, 1], [0.1, 0.2, 0.5, 0.5, 8]])

    # lables=torch.tensor([1,0.2,0.2,0.5,0.5]).reshape(1,-1)

    # corr=process_iou(detections,lables,torch.tensor([0.2]))
    # print(corr)
