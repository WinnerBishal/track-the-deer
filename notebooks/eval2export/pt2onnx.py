import ultralytics
from model_paths import all_models

for model_path in all_models:
    model = ultralytics.YOLO(model_path)
    model.export(format = 'onnx', opset = 13, simplify = True, half = True, nms = True, batch = 16, device = 0)


