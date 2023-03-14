import os
import cv2
import numpy as np


class BeltTransform:
    def __init__(self, root_path, p_rect, scales):
        self.root_path = root_path
        self.p_rect = np.array(p_rect)
        self.h_old, self.w_old = 0, 0
        self.h_new, self.w_new, self.matrix = self._getMatrix(p_rect, scales)
        self._makeDir()

    @staticmethod
    def _getMatrix(p, s):
        w = int((p[3][0] - p[2][0] + 1) * s[0])
        h = int((p[2][1] - p[0][1] + 1) * s[1])
        fig = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(np.float32(p), fig)
        return h, w, matrix

    def _makeDir(self):
        if self.root_path:
            if os.path.isdir(self.root_path):
                root_path_belt = self.root_path.replace('ImageAndLabel', 'BeltImageAndLabel')
                if not os.path.exists(root_path_belt):
                    os.mkdir(root_path_belt)
                img_dir = os.path.join(root_path_belt, 'images')
                txt_dir = os.path.join(root_path_belt, 'labels')
                if not os.path.exists(img_dir):
                    os.mkdir(img_dir)
                    os.mkdir(txt_dir)
            if os.path.isfile(self.root_path):
                save_path = self.root_path.replace('SplitVideo', 'BeltVideo')
                if not os.path.exists(os.path.dirname(save_path)):
                    os.mkdir(os.path.dirname(save_path))

    def allImage(self):
        img_dir = os.path.join(self.root_path, 'images')
        txt_dir = os.path.join(self.root_path, 'labels')
        img_list = os.listdir(img_dir)
        image = cv2.imread(os.path.join(img_dir, img_list[0]), 1)
        self.h_old, self.w_old, _ = image.shape
        for img_name in img_list:
            txt_name = img_name.split('.')[0] + '.txt'
            img_path = os.path.join(img_dir, img_name)
            txt_path = os.path.join(txt_dir, txt_name)
            if os.path.exists(txt_path):
                image = cv2.imread(img_path, 1)
                xywh = np.loadtxt(txt_path).reshape((-1, 5))[:, 1:]
                img_belt, txt_belt = self._rectImage(image, xywh)
                img_path_belt = img_path.replace('ImageAndLabel', 'BeltImageAndLabel')
                txt_path_belt = txt_path.replace('ImageAndLabel', 'BeltImageAndLabel')
                cv2.imwrite(img_path_belt, img_belt)
                if os.path.exists(txt_path_belt):
                    os.remove(txt_path_belt)
                for xywh in txt_belt:
                    xywh = xywh.tolist()  # normalized xywh
                    line = (0, *xywh)  # label format
                    with open(txt_path_belt, 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

    def oneImage(self, img_path, txt_path):
        image = cv2.imread(img_path, 1)
        self.h_old, self.w_old, _ = image.shape
        xywh = np.loadtxt(txt_path).reshape((-1, 5))[:, 1:]
        return self._rectImage(image, xywh)

    def videoTransform(self, view):
        cap = cv2.VideoCapture(self.root_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        save_path = self.root_path.replace('SplitVideo', 'BeltVideo')
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (self.w_new, self.h_new))
        flag, img = cap.read()
        while flag:
            img_belt = cv2.warpPerspective(img, self.matrix, (self.w_new, self.h_new))
            vid_writer.write(img_belt)
            if view:
                cv2.imshow('old', img)
                cv2.imshow('new', cv2.resize(img_belt, (self.w_new // 2, self.h_new // 2)))
                cv2.waitKey(10)
            flag, img = cap.read()

    def onlyImage(self, img):
        return cv2.warpPerspective(img, self.matrix, (self.w_new, self.h_new))

    def _rectImage(self, img, xywh):
        xyxy = self._xywh2xyxy(xywh)
        boxes = self._xyxy2boxes(xyxy)
        img_belt = cv2.warpPerspective(img, self.matrix, (self.w_new, self.h_new))
        boxes_belt = self._boxes2boxes(boxes)
        xyxy_belt = self._boxes2xyxy(boxes_belt)
        xywh_belt = self._xyxy2xywh(xyxy_belt)
        for box in boxes_belt:
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[2]), int(box[3]))
            cv2.rectangle(img_belt, p1, p2, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        return img_belt, xywh_belt

    def _boxes2boxes(self, p):
        pp = np.empty_like(p, dtype=int)
        pp[:, [0, 2]] = (self.matrix[0][0] * p[:, [0, 2]] + self.matrix[0][1] * p[:, [1, 3]] + self.matrix[0][2]) / (
            (self.matrix[2][0] * p[:, [0, 2]] + self.matrix[2][1] * p[:, [1, 3]] + self.matrix[2][2]))
        pp[:, [1, 3]] = (self.matrix[1][0] * p[:, [0, 2]] + self.matrix[1][1] * p[:, [1, 3]] + self.matrix[1][2]) / (
            (self.matrix[2][0] * p[:, [0, 2]] + self.matrix[2][1] * p[:, [1, 3]] + self.matrix[2][2]))
        return pp

    @staticmethod
    def _xywh2xyxy(x):
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def _xyxy2boxes(self, xyxy):
        boxes = np.empty_like(xyxy, dtype=int)
        boxes[:, [0, 2]] = np.round(xyxy[:, [0, 2]] * self.w_old)  # x1, x2
        boxes[:, [1, 3]] = np.round(xyxy[:, [1, 3]] * self.h_old)  # y1, y2
        return boxes

    def _boxes2xyxy(self, boxes):
        xyxy = np.empty_like(boxes, dtype=float)
        xyxy[:, [0, 2]] = boxes[:, [0, 2]] / self.w_new  # x1, x2
        xyxy[:, [1, 3]] = boxes[:, [1, 3]] / self.h_new  # y1, y2
        return xyxy

    @staticmethod
    def _xyxy2xywh(x):
        y = np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2
        y[:, 2] = x[:, 2] - x[:, 0]
        y[:, 3] = x[:, 3] - x[:, 1]
        return y


if __name__ == '__main__':
    img_path = '../202-233'
    save_path = '../202-233/change'
    # img1 = cv2.imread(img1_path, 1)
    po_202_233 = [[635, 0], [940, 0], [133, 1079], [1412, 1079]]
    po_103_102 = [[518, 200], [882, 200], [-355, 719], [1841, 719]]
    imgs_list = os.listdir(img_path)  # 列出图目录下所有的图片

    A = BeltTransform(None, po_202_233, (0.5, 1.5))
    for i in imgs_list:
        img_dir = os.path.join(img_path, i)  # 保存处理后的数据(图片+标签)的文件夹
        save_dir = os.path.join(save_path, i)
        img1 = cv2.imread(img_dir)
        img2 = A.onlyImage(img1)
        cv2.imshow('wwa', img2)
        cv2.imwrite(save_dir,img2)
        cv2.waitKey(1)
    # cv2.imwrite('../1t.png', img2)
