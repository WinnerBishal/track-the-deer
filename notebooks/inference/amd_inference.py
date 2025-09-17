import os
import glob
import time
import math
import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
import json
import copy
import csv
# import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
# ─── Setup ────────────────────────────────────────────────────────────────────
IMG_DIR   = "../data/Lila.v1.3k/valid/images"
LABEL_DIR = "../data/Lila.v1.3k/valid/labels"
MODEL_PATH = "../onnx_models_op13/"
MODELS = os.listdir(MODEL_PATH)

# --------------------------------HELPER FUNCTIONS------------------------------

def prepare_GTs(img_dir = IMG_DIR, label_dir = LABEL_DIR):

    GTs = dict() # Ground truths
    
    image_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    n_boxes = 0
    for img_path in image_paths:

        boxes = [] # boxes with used? flag

        img = cv2.imread(img_path)
        orig_h, orig_w = img.shape[:2]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, stem + ".txt")

        with open(label_path) as f:
            for line in f:
                cls, x_n, y_n, w_n, h_n = line.strip().split()
                cls = int(cls)
                cx = float(x_n) * orig_w
                cy = float(y_n) * orig_h
                bw = float(w_n) * orig_w
                bh = float(h_n) * orig_h
                x1, y1 = cx - bw/2, cy - bh/2
                x2, y2 = cx + bw/2, cy + bh/2
                boxes.append({"box": [int(x1), int(y1), int(x2), int(y2)], "used": False})
                n_boxes += 1
        GTs[f'{stem}'] = boxes

    return GTs, n_boxes


import torch # Required for CUDA synchronization

def batch_inference(model = MODEL_PATH + MODELS[5], img_dir = IMG_DIR, batch_size = 16):
    
    # Prioritize providers for best performance
    providers = [
        'CUDAExecutionProvider',
        'CPUExecutionProvider'
    ]
    print(f"Performing inference with {model} using providers: {providers}")

    # Use SessionOptions to enable optimizations
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(model, sess_options=sess_options, providers=providers)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape # e.g., [1, 3, 640, 640]
    input_shape[2] = 640
    input_shape[3] = 640

    image_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    num_batches = math.ceil(len(image_paths) / batch_size)
    PBs = dict()

    # --- 1. Warmup Phase ---
    print("Warming up the model...")
    dummy_input = np.zeros((batch_size, 3, 640, 640), dtype=np.float16)
    for _ in range(3):
        _ = session.run(None, {input_name: dummy_input})
    print("Warmup complete.")

    # --- 2. Benchmarking Phase ---
    total_preproc_time = 0.0
    total_postproc_time = 0.0

    # Start timer for the whole inference process
    torch.cuda.synchronize() # Ensure GPU is idle before starting
    start_time = time.perf_counter()

    for i in tqdm(range(num_batches), desc="Batch Inference"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(image_paths))
        batch_paths = image_paths[start_idx:end_idx]
        
        # --- Preprocessing ---
        preproc_start = time.perf_counter()
        batch_input = []
        original_dims = []
        for img_path in batch_paths:
            img = cv2.imread(img_path)
            orig_h, orig_w = img.shape[:2]
            original_dims.append((orig_h, orig_w))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Use np.float32 for compatibility and performance
            resized = cv2.resize(img, (640, 640)).astype(np.float16) / 255.0
            inp = resized.transpose(2, 0, 1)
            batch_input.append(inp)
        batch_input_np = np.stack(batch_input)
        total_preproc_time += time.perf_counter() - preproc_start
        
        # --- Inference ---
        batch_outputs = session.run(None, {input_name: batch_input_np})

        # --- Postprocessing ---
        postproc_start = time.perf_counter()
        for j, single_output in enumerate(batch_outputs[0]):
            image_path = batch_paths[j]
            stem = os.path.splitext(os.path.basename(image_path))[0]
            orig_h, orig_w = original_dims[j]
            
            # (Your existing postprocessing logic)
            preds = single_output
            boxes = preds[:, :4]
            scores = preds[:, 4]
            mask = scores > 0
            boxes = boxes[mask]
            scores = scores[mask]
            w_ratio = orig_w / input_shape[3]
            h_ratio = orig_h / input_shape[2]
            preds_list = []
            for k, box in enumerate(boxes):
                if scores[k] > 0.25: # Example confidence threshold
                    x1 = box[0] * w_ratio
                    y1 = box[1] * h_ratio
                    x2 = box[2] * w_ratio
                    y2 = box[3] * h_ratio
                    preds_list.append({
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "conf": scores[k]
                    })
            PBs[stem] = preds_list
        total_postproc_time += time.perf_counter() - postproc_start

    # Stop timer after all batches are processed and GPU is finished
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    # --- 3. Calculate Final Timings ---
    total_inference_time = end_time - start_time
    # Subtract pre/post processing time to isolate model inference
    pure_inf_time = total_inference_time - total_preproc_time - total_postproc_time
    
    # Average time per image
    avg_inf_time_ms = (pure_inf_time / len(image_paths)) * 1000
    avg_preproc_time_ms = (total_preproc_time / len(image_paths)) * 1000
    avg_postproc_time_ms = (total_postproc_time / len(image_paths)) * 1000
    
    print(f"\n--- Benchmark Results ---")
    print(f"Total images: {len(image_paths)}")
    print(f"Pure inference time: {pure_inf_time:.3f} s")
    print(f"Average inference time per image: {avg_inf_time_ms:.3f} ms")
    print(f"Average preprocessing time per image: {avg_preproc_time_ms:.3f} ms")
    print(f"Average postprocessing time per image: {avg_postproc_time_ms:.3f} ms")
    print("-------------------------")

    return PBs, [avg_inf_time_ms, avg_preproc_time_ms, avg_postproc_time_ms]

def calculate_iou_bbox(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    interA = interW * interH
    areaA = (boxA[2]-boxA[0]+1)*(boxA[3]-boxA[1]+1)
    areaB = (boxB[2]-boxB[0]+1)*(boxB[3]-boxB[1]+1)
    return interA / float(areaA + areaB - interA + 1e-6)


def calculate_counts(GTs, PBs, iou_threshold = 0.5):
    
    final_list = []
    
    GTs_copy = copy.deepcopy(GTs)

    for stem in tqdm(PBs.keys(), desc = "Matching Predictions"):
        
        preds = PBs[stem]
        gtbs = GTs_copy[stem]


        preds.sort(key = lambda x: x["conf"], reverse = True)

        for pred_dict in preds:
            best_iou = 0
            best_gt_idx = -1

            for i, gtb_dict in enumerate(gtbs):
                iou = calculate_iou_bbox(gtb_dict["box"], pred_dict["box"])
                # print(f"IOU: {iou}, gtb: {gtb_dict['box']}, predb : {pred_dict['box']}")
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            if best_iou > iou_threshold and not gtbs[best_gt_idx]["used"]:
                final_list.append({"conf": pred_dict["conf"], "type": "TP"})
                gtbs[best_gt_idx]["used"] = True
            else:
                final_list.append({"conf": pred_dict["conf"], "type": "FP"})
    
    return final_list

def process_fptpconflist(fp_tp_conf_list, n_boxes):

    fp_tp_conf_list.sort(key = lambda x: x["conf"], reverse = True)

    tp = 0
    fp = 0

    rcs = []
    prs = []

    fp_dot5 = 0
    tp_dot5 = 0

    for pred in fp_tp_conf_list:
        if pred["type"] == "FP":
            fp += 1

            if pred["conf"] >= 0.5:
                fp_dot5 += 1

        else:
            tp += 1
            if pred["conf"] >= 0.5:
                tp_dot5 += 1
    
        recall = tp / (n_boxes + 1e-6)
        precision = tp / (fp + tp + 1e-6)

        rcs.append(recall)
        prs.append(precision)
    
    recall_dot5 = tp_dot5/ (n_boxes + 1e-6)
    precision_dot5 = tp_dot5 / (fp_dot5 + tp_dot5 + 1e-6)
    f1_dot5 = 2 * precision_dot5 * recall_dot5 / (precision_dot5 + recall_dot5 + 1e-6)

    rcs = np.array([0.] + rcs)
    prs = np.array([1.] + prs)

    ap = np.trapezoid(prs, rcs)

    # plt.figure(figsize=(10, 8))
    # plt.title("Precision - Recall Curve")
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.plot(rcs, prs, marker = '.', label = f'AP : {ap:.3f}')

    return ap, prs, rcs, [precision_dot5, recall_dot5, f1_dot5]

GTs, n_boxes = prepare_GTs()
outputs = []

for model in MODELS:

    PBs, times = batch_inference(model = MODEL_PATH + model)
    fp_tp_conf_list = calculate_counts(GTs, PBs)
    ap, _, _, metrics_dot5 = process_fptpconflist(fp_tp_conf_list, n_boxes)

    output = {
	      'model' : f'{os.path.splitext(model)[0]}',
	      'inference_time' : times[0],
          'preprocessing_time' : times[1],
          'postprocessng_time' : times[2],
	    #   'tp' : total_tp,
	    #   'fp' : total_fp,
	    #   'fn' : total_fn,
	      'precision' : metrics_dot5[0],
	      'recall' : metrics_dot5[1],
	      'f1_score' : metrics_dot5[2],
          'AP@0.5' : ap
	    }
    outputs.append(output)
    # output_file = f'{os.path.splitext(model)[0]}.json'
    # with open(output_file, 'w') as f:
    #     json.dump(output, f, indent=4)

out_csv = 'amd_cpu_eval.csv'
headers = outputs[0].keys()
with open(out_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(outputs)
# PBs, inf_time = batch_inference()
# fp_tp_conf_list = calculate_counts(GTs, PBs)

