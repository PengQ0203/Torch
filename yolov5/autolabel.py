import os
import cv2
import sys
import argparse
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from baseui import UiMainWindow
from pathlib import Path
from utils.general import print_args, xyxy2xywh, xywh2xyxy
from labeldetect import Detect

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def xyxy2boxes(xyxy, shape):
    boxes = np.empty_like(xyxy, dtype=int)
    boxes[:, [0, 2]] = np.round(xyxy[:, [0, 2]] * shape[1])  # x1, x2
    boxes[:, [1, 3]] = np.round(xyxy[:, [1, 3]] * shape[0])  # y1, y2
    return boxes


class ShowUi(UiMainWindow):
    def __init__(self):
        super(ShowUi, self).__init__()
        self.count = 0
        self.frames = 0
        self.cap = None
        self.im0 = None
        self.det = None
        self.path_list = None
        opt = parse_opt()
        self.detect = Detect(**vars(opt))

    def open_picture(self):
        self.count = 0
        if self.isvideo:
            self.cap = cv2.VideoCapture(self.path)
            self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            img, self.im0, self.det = self.rect_video()
            img_show = self.resize_image(img)
        else:
            self.path_list = os.listdir(os.path.join(self.path, 'images'))
            self.frames = len(self.path_list)
            img = self.rect_image(self.count)
            img_show = self.resize_image(img)
        self.count += 1
        self.show_fps()
        self.label_image.setPixmap(QPixmap.fromImage(img_show))

    def auto_video(self):
        if self.isvideo:
            if self.auto_label:
                self.auto_label = False
                self.timer_video.stop()
                self.button_auto.setText("开启自动")
            elif self.cap and self.count < self.frames:
                self.timer_video.start(30)
                self.auto_label = True
                self.button_auto.setText("暂停自动")

    def next_image(self):
        if self.path_list and self.count < self.frames:
            img = self.rect_image(self.count)
            img_show = self.resize_image(img)
            self.count += 1
            self.show_fps()
            self.label_image.setPixmap(QPixmap.fromImage(img_show))

    def next_video(self):
        gn = np.array(self.im0.shape)[[1, 0, 1, 0]]
        name = str(self.count - 1).zfill(8)
        img_name = name + '.png'
        txt_name = name + '.txt'
        cv2.imwrite(os.path.join(self.save_img_dir, img_name), self.im0)
        txt_path = os.path.join(self.save_txt_dir, txt_name)
        if os.path.exists(txt_path):
            os.remove(txt_path)
        for xyxy in self.det:
            xywh = (xyxy2xywh(xyxy.reshape(1, 4)) / gn).reshape(-1).tolist()  # normalized xywh
            line = (0, *xywh)  # label format
            with open(txt_path, 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')
        if self.cap and self.count < self.frames:
            img, self.im0, self.det = self.rect_video()
            if not (img is False):
                img_show = self.resize_image(img)
                self.label_image.setPixmap(QPixmap.fromImage(img_show))
            else:
                print("文件错误，无法读取")
            self.count += 1
            self.show_fps()
        elif self.timer_video.isActive():
            self.timer_video.stop()

    def last_image(self):
        if self.path_list and self.count > 0:
            img = self.rect_image(self.count - 2)
            img_show = self.resize_image(img)
            self.count -= 1
            self.show_fps()
            self.label_image.setPixmap(QPixmap.fromImage(img_show))

    def last_video(self):
        pass

    def remove_image(self):
        if not self.isvideo:
            img_name = self.path_list[self.count - 1]
            txt_name = img_name.split('.')[0] + '.txt'
            img_path = os.path.join(self.path, 'images', img_name)
            txt_path = os.path.join(self.path, 'labels', txt_name)
            os.remove(img_path)
            os.remove(txt_path)
            self.next_image()

    def nosave_video(self):
        if self.cap and self.count < self.frames:
            img, self.im0, self.det = self.rect_video()
            if not (img is False):
                img_show = self.resize_image(img)
                self.label_image.setPixmap(QPixmap.fromImage(img_show))
            else:
                print("文件错误，无法读取")
            self.count += 1
            self.show_fps()

    def show_fps(self):
        self.label_fps.setText(f"帧数：{self.count}/{self.frames}")

    def resize_image(self, img):
        label_size = self.label_image.size()
        scale = min(label_size.width() / img.shape[1], label_size.height() / img.shape[0])
        w_new, h_new = int(scale * img.shape[1]), int(scale * img.shape[0])
        # img = cv2.resize(img, (label_size.width(), label_size.height()))
        img = cv2.resize(img, (w_new, h_new))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_height, img_width, channels = img.shape
        bytesPerLine = channels * img_width
        return QImage(img.data, img_width, img_height, bytesPerLine, QImage.Format_RGB888)

    def rect_video(self):
        flag, img = self.cap.read()
        if flag:
            im0 = img.copy()
            det = self.detect.run(img).cpu().numpy()
            if len(det):
                for i, box in enumerate(det):
                    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                    cv2.rectangle(img, p1, p2, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                return img, im0, det
            else:
                self.count += 1
                if self.count < self.frames:
                    return self.rect_video()
                else:
                    return False, False, False
        else:
            return False, False, False

    def rect_image(self, count):
        img_name = self.path_list[count]
        txt_name = img_name.split('.')[0] + '.txt'
        img_path = os.path.join(self.path, 'images', img_name)
        txt_path = os.path.join(self.path, 'labels', txt_name)
        if os.path.exists(txt_path):
            image = cv2.imread(img_path, 1)
            xywh = np.loadtxt(txt_path).reshape((-1, 5))[:, 1:]
            xyxy = xywh2xyxy(xywh)
            boxes = xyxy2boxes(xyxy, image.shape)
            for i, box in enumerate(boxes):
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                cv2.rectangle(image, p1, p2, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            return image
        else:
            os.remove(img_path)
            self.count += 1
            if self.count <= self.frames:
                return self.rect_image(self.count)
            else:
                return False


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'BigCoal/exp8/weights/best.pt',
                        help='model path(s)')
    parser.add_argument('--data', type=str, default=ROOT / 'script/dataset.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = ShowUi()
    ui.show()
    sys.exit(app.exec_())
