from ultralytics import YOLO
import cv2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI environments

import os


model = YOLO("../experiments/Lila.v1.3k_y10s_ep100/weights/best.pt")

# test_img_path = '../data/raw/deer-images-lila-cam-trap/'
test_img_path = '../data/Lila.v1.3k/valid/images/'


# Draw random 5 images from the test image path
test_images = os.listdir(test_img_path)
test_images = np.random.choice(test_images, 5, replace=False)

plt.figure(figsize=(10, 20))

for img in test_images:
    img_path = os.path.join(test_img_path, img)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(image)

    # Plot the results
    plt.imshow(results[0].plot())
    plt.axis('off')
    plt.savefig(f'{img}_inference_y10s_Lila3k.png') 
 

