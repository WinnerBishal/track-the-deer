import yaml
import argparse
import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import ultralytics as ultics


best_model_dir = "../experiments/Lila.v1.3k_y10s_ep100/weights/best.pt"

model = ultics.YOLO(best_model_dir)
data_config = "../data/Lila.v1.3k/data.yaml"

results = model.val(data=data_config,
                    split='val',
                    save_json=True,  
                    save_txt=True,   
                    save_conf=True,  
                    save_hybrid=True,  
                    plots=True)  

metrics = {
            'model_path': best_model_dir,
            'experiment_name': Path(best_model_dir).parent.parent.name,
            'precision': results.box.mp if hasattr(results.box, 'mp') else 0.0,
            'recall': results.box.mr if hasattr(results.box, 'mr') else 0.0,
            'map50': results.box.map50 if hasattr(results.box, 'map50') else 0.0,
            'map50_95': results.box.map if hasattr(results.box, 'map') else 0.0,
            'fitness': results.fitness if hasattr(results, 'fitness') else 0.0,
        }

result_df = pd.DataFrame([metrics])

# Export results to CSV
output_csv = f'{metrics["experiment_name"]}_test.csv'
result_df.to_csv(output_csv, index=False)

print(result_df)
