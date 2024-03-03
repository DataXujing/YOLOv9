import argparse
import os
import platform
import sys
from pathlib import Path
import cv2
import numpy as np

import torch

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams, letterbox
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
# from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


test_path = "./datasets/xray/images/test"
weights = "runs/train/exp/weights/last.pt"
dnn=False,  # use OpenCV DNN for ONNX inference
data = "./data/xray.yaml"
half = False
imgsz=(640, 640)

# 后处理的一些参数
conf_thres = 0.25
iou_thres = 0.45
classes = None # filter class
agnostic_nms = False # class-agnostic NMS
max_det = 1000 # maximum detections per image

colors = [(139,0,139),(255,228,225),(0,100,0),(0,255,255),(0,0,255),(0,255,0),(255,0,0),(255,191,0),
    (139,139,0),(255,111,131),(147,20,255),(0,127,255),
    (205,89,105),(255,0,255),(205,82,180),(0,0,139),(144,238,144),(255,191,0)]

# 模型加载
device = select_device(0)   #0，1，2，3 or cpu
print("device: ",device)
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)  # check image size

imgs = os.listdir(test_path)

for file in imgs:
    img_path = os.path.join(test_path,file)
    img = cv2.imread(img_path)

    # 前处理
    im = letterbox(img, imgsz, stride=stride, auto=pt)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).to(device)

    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # pred
    pred = model(im, augment=False, visualize=False)[0]
    # print(pred)

    # 后处理
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    # Process predictions
    for i, det in enumerate(pred):  # per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label= f'{names[c]} {conf:.2f}'


                tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
                color = colors[c]
                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))

                cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(label, 0, fontScale=tl / 2, thickness=tf)[0]
                font_size = 20

                if (c1[1] - t_size[1] - 3) > 0:
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    # c2 = (c1[0] + len(label)), c1[1] - t_size[1] - 10

                    cv2.rectangle(img, (int(c1[0]-tl/2),c1[1]), c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 2, [0, 0, 0], 
                        thickness=tf, lineType=cv2.LINE_AA)
                else:
                    c2 = c1[0] + t_size[0], c1[1] + t_size[1] + 3
                    # c2 = c1[0] + len(label), c1[1] + t_size[1] + 10
                    cv2.rectangle(img, (int(c1[0]-tl/2),c1[1]), c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, label, (c1[0], c2[1] - 2), 0, tl / 2, [0, 0, 0], 
                        thickness=tf, lineType=cv2.LINE_AA)
    # cv2.imwrite()
    if not os.path.exists("res"):
        os.makedirs("res")
    cv2.imencode(".jpg",img)[1].tofile(os.path.join("res",file)) 
    print(file)
                   












     



