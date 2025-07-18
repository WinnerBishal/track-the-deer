import numpy as np
import os

annotations_dir = '../data/Lila.v1.3k/train/labels/'
images_source_dir = '../data/raw/deer-images-lila-cam-trap/'
images_target_dir = '../data/Lila.v1.3k/train/images/'

images = os.listdir(images_source_dir)
images = [img for img in images if img.endswith('.jpg') ]
annots = os.listdir(annotations_dir)
annots = [annot for annot in annots if annot.endswith('.txt')]

annot_names = [os.path.splitext(annot)[0] for annot in annots]
img_names = [os.path.splitext(img)[0] for img in images]

images_to_keep = [img for img in img_names if img in annot_names]

# copy images to target directory
for img in images_to_keep:
    source_path = os.path.join(images_source_dir, img + '.jpg')
    target_path = os.path.join(images_target_dir, img + '.jpg')
    if not os.path.exists(target_path):
        os.system(f'cp {source_path} {target_path}')

# Split the images into train and val sets
np.random.seed(42)
np.random.shuffle(images_to_keep)
split_index = int(0.8 * len(images_to_keep))
train_images = images_to_keep[:split_index]
val_images = images_to_keep[split_index:]

train_imgdir = images_target_dir
val_imgdir = '../data/Lila.v1.3k/valid/images/'

train_labels_dir = annotations_dir
val_labels_dir = '../data/Lila.v1.3k/valid/labels/'

for img in val_images:
    source = os.path.join(train_imgdir, img + '.jpg')
    target = os.path.join(val_imgdir, img + '.jpg')
    if not os.path.exists(target):
        os.system(f'mv {source} {target}')
        print(f'{target} {os.path.exists(target)}')

for annot in val_images:
    source = os.path.join(train_labels_dir, annot + '.txt')
    target = os.path.join(val_labels_dir, annot + '.txt')
    if not os.path.exists(target):
        os.system(f'mv {source} {target}')
        print(f'{target} {os.path.exists(target)}')
