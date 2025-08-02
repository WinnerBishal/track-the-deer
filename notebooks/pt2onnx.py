import numpy as np
import ultralytics
import onnx
import os
import shutil

yolo10n = "../experiments/Lila.v1.3k_y10n_ep100/weights/best.pt"
yolo11n = "../experiments/Lila.v1.3k_y11_ep100/weights/best.pt"
yolo8n = "../experiments/Lila.v1.3k_y8_ep100/weights/best.pt"

yolo9s = "../experiments/Lila.v1.3k_y9s_ep100/weights/best.pt"
yolo10s = "../experiments/Lila.v1.3k_y10s_ep100/weights/best.pt"
yolo11s = "../experiments/Lila.v1.3k_y11s_ep100/weights/best.pt"

yolo11m = "../experiments/Lila.v1.3k_y11m_ep100/weights/best.pt"
yolo11x = "../experiments/Lila.v1.3k_y11x_ep100/weights/best.pt"

all_models = [yolo10n, yolo11n, yolo8n, yolo9s, yolo10s, yolo11s, yolo11m, yolo11x]

for model in all_models:    
    model = ultralytics.YOLO(model)
    model.export(format = 'onnx', opset = 11, simplify = True, dynamic = True, half = True, nms = True, device = 0)


