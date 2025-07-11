from ultralytics import YOLO
import cv2

import numpy as np
import matplotlib.pyplot as plt

import os


model = YOLO("../experiments/7-8-25_y11_exp1_3/weights/best.pt")

test_img_path = '../data/raw/deer-images-lila-cam-trap/'

# Draw random 5 images from the test image path
test_images = os.listdir(test_img_path)
test_images = np.random.choice(test_images, 5, replace=False)

for img in test_images:
    img_path = os.path.join(test_img_path, img)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(image)

    # Plot the results
    plt.subplot(5, 1, test_images.tolist().index(img) + 1)
    plt.imshow(results[0].plot())
    plt.axis('off')
plt.tight_layout()
plt.show()  
