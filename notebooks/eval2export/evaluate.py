'''
This is the main script for evaluating all trained models in a single run. 
Add directory to newly trained models here for extended evaluation.
It writes the results from evaluation to a csv file.
'''

import ultralytics as ultics
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from model_paths import all_models

def evaluate(all_models, data_config = "data/Lila.v1.3k/data.yaml"):
    results = []
    for model_path in tqdm(all_models, desc="Evaluating models"):
        model = ultics.YOLO(model_path)
        result = model.val(data=data_config,
                           split='val',
                           save_json=True,  
                           save_txt=True,   
                           save_conf=True,  
                           save_hybrid=True,  
                           plots=True) 
        
        metrics = {
            'experiment_name': Path(model_path).parent.parent.name,
            'model_parameters': model.info()[1],
            'GFLOPs': model.info()[3],
            'precision': result.box.mp,
            'recall': result.box.mr,
            'F1' : result.box.f1[0],
            'map50': result.box.map50,
            'map50_95': result.box.map,
            'fitness': result.fitness,
            'preprocessing_time': result.speed['preprocess'],
            'inference_time': result.speed['inference'],
            'postprocessing_time': result.speed['postprocess'],
            'time_loss': result.speed['loss'],
            'total_time': result.speed['inference'] + result.speed['postprocess'] + result.speed['loss'] + result.speed['preprocess'],
        }
        results.append(metrics)
    # Truncate values to 4 decimal places
    for result in results:
        for key, value in result.items():
            if isinstance(value, float):
                result[key] = round(value, 4)
    pd.DataFrame(results).to_csv('evaluation_results.csv', index=True)
    print(pd.DataFrame(results))

if __name__ == "__main__":
    evaluate(all_models)

# Uncomment the following code to visualize true vs predicted bounding boxes for a specific image
'''
print("Working")
import yaml
import cv2
import matplotlib.pyplot as plt
with open('../../runs/detect/val12/predictions.json', 'r') as f:
    results = yaml.safe_load(f)

img_id = 'loc_0088_im_001827'
pred_boxes = []

for result in results:
    if result['image_id'] == img_id and result['score'] > 0.5:
        pred_boxes.append(result['bbox'])

print(pred_boxes)

true_label_dir = '../../data/Lila.v1.3k/valid/labels'
with open(f'{true_label_dir}/{img_id}.txt', 'r') as f:
    true_boxes = [line.strip().split()[1:] for line in f.readlines()]

img = cv2.imread(f'../../data/Lila.v1.3k/valid/images/{img_id}.jpg')

for box in pred_boxes:
    # Convert YOLO prediction format to bounding box coordinates
    x_corner, y_corner, width, height = map(float, box)
    x1 = int((x_corner ) )
    y1 = int((y_corner ))
    x2 = int((x_corner + width ) )
    y2 = int((y_corner + height) )
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
    
print(true_boxes)
    
for box_points in true_boxes:
    x_center, y_center, width, height = map(float, box_points)
    x1 = int((x_center - width / 2) * img.shape[1])
    y1 = int((y_center - height / 2) * img.shape[0])
    x2 = int((x_center + width / 2) * img.shape[1])
    y2 = int((y_center + height / 2) * img.shape[0])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.savefig('true_v_pred_bbox.png')

'''