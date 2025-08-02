'''
Run this script after running pt_to_onnx.py to move and rename the ONNX files.
This script will copy the ONNX files to a specified directory and rename them according to the model names.
'''

import os
import shutil

# Define the paths to the best.pt files
model_paths = {
    "yolo10n": "../experiments/Lila.v1.3k_y10n_ep100/weights/best.pt",
    "yolo11n": "../experiments/Lila.v1.3k_y11_ep100/weights/best.pt",
    "yolo8n": "../experiments/Lila.v1.3k_y8_ep100/weights/best.pt",
    "yolo9s": "../experiments/Lila.v1.3k_y9s_ep100/weights/best.pt",
    "yolo10s": "../experiments/Lila.v1.3k_y10s_ep100/weights/best.pt",
    "yolo11s": "../experiments/Lila.v1.3k_y11s_ep100/weights/best.pt",
    "yolo11m": "../experiments/Lila.v1.3k_y11m_ep100/weights/best.pt",
    "yolo11x": "../experiments/Lila.v1.3k_y11x_ep100/weights/best.pt",
}

destination_dir = "onnx_models"
os.makedirs(destination_dir, exist_ok=True)

for model_name, pt_path in model_paths.items():
    onnx_path = pt_path.replace("best.pt", "best.onnx")

    if os.path.exists(onnx_path):
        new_filename = f"{model_name}.onnx"
        destination_path = os.path.join(destination_dir, new_filename)

        try:
            shutil.copy(onnx_path, destination_path)
            print(f"Copied {onnx_path} to {destination_path}")
        except FileNotFoundError:
            print(f"Error: {onnx_path} not found.")
    else:
        print(f"Warning: {onnx_path} does not exist.")

print("Process completed.")
