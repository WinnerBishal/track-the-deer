import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

img_dir = '../data/Deer.v6i.y11/train/images/'
img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

# Draw random 1 image from the test image path
test_images = np.random.choice(img_files, 5, replace=False)

# draw corresponding labels
label_dir = '../data/Deer.v6i.y11/train/labels/'

img_labels = []
for img in test_images:
    label_file = os.path.join(label_dir, img.replace('.jpg', '.txt'))
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            labels = f.readlines()
        img_labels.append((img, labels))
    else:
        img_labels.append((img, []))

# plt.figure(figsize=(10, 20))


# Draw annotaions on images
for i, (img, labels) in enumerate(img_labels):
    img_path = os.path.join(img_dir, img)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw labels on the image
    for label in labels:
        parts = label.strip().split()
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])
        
        # Convert YOLO format to bounding box coordinates
        img_height, img_width, _ = image.shape
        x1 = int((x_center - width / 2) * img_width)
        y1 = int((y_center - height / 2) * img_height)
        x2 = int((x_center + width / 2) * img_width)
        y2 = int((y_center + height / 2) * img_height)

        # Draw rectangle and class label
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, f'Class {class_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Plot the results

        plt.imshow(image)
        
        plt.axis('off')
        plt.savefig(f'{i}_annotated_images_dv6.png')

