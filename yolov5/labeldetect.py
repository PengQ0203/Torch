import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords, xyxy2xywh, check_img_size


class Detect:
    def __init__(
            self,
            weights,
            data,
            imgsz,
            conf_thres=0.25,  # 检测置信度阈值，单值不同时应为列表
            iou_thres=0.45,  # 检测iou阈值，应为列表
            device='0',  # cuda device, i.e. 0 or 1 or ... or cpu
            max_det=1000,
            line_thickness=3,
            half=True,  # 以FP16运行
    ):
        # load configs
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.line_thickness = line_thickness
        # Detect Load model
        self.device = select_device(device)

        assert os.path.exists(weights), 'The Detect model file does not exist'
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False, data=data, fp16=half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = (640, 640)  # check image size
        self.model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup

    @torch.no_grad()
    def run(self, im):
        im, im0s = self._pre_process(im)
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False, max_det=self.max_det)

        det = pred[0]
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()

        return det[:, :4]

    def _pre_process(self, im):
        img0 = im.copy()
        img = letterbox(im, self.imgsz, stride=self.stride, auto=self.pt)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        return img, img0
