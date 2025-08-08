'''
Run this script after running pt_to_onnx.py to move and rename the ONNX files.
This script will copy the ONNX files to a specified directory and rename them according to the model names.
'''

import os
import shutil
from model_paths import all_models

destination_dir = "onnx_models_op15"
os.makedirs(destination_dir, exist_ok=True)

for pt_path in all_models:
    # Extract experiment directory, e.g., Lila.v1.3k_y10n_ep100
    exp_dir = os.path.basename(os.path.dirname(os.path.dirname(pt_path)))
    # Extract short model name, e.g., y10n
    short_name = exp_dir.split('_')[-2] if len(exp_dir.split('_')) > 1 else exp_dir
    onnx_path = pt_path.replace("best.pt", "best.onnx")

    if os.path.exists(onnx_path):
        new_filename = f"{short_name}.onnx"
        destination_path = os.path.join(destination_dir, new_filename)

        try:
            shutil.copy(onnx_path, destination_path)
            print(f"Copied {onnx_path} to {destination_path}")
        except FileNotFoundError:
            print(f"Error: {onnx_path} not found.")
    else:
        print(f"Warning: {onnx_path} does not exist.")

print("Process completed.")
