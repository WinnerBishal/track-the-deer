import xml.etree.ElementTree as ET
import os, shutil

xml_folder = '../data/raw/annotations/annotations/'
image_folder = '../data/raw/deer-images-lila-cam-trap/'

output_labels = '../data/Deer.v6i.y11/test/labels/'
os.makedirs(output_labels, exist_ok=True)

output_images = '../data/Deer.v6i.y11/test/images/'
os.makedirs(output_images, exist_ok=True)

# Open xml file
xml_files = [f for f in os.listdir(xml_folder) if f.endswith('.xml')]
print(f"Found {len(xml_files)} XML files in {xml_folder}")
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
print(f"Found {len(image_files)} image files in {image_folder}")

# Save corresponding images to output folder
for xml_file in xml_files:
    img_name = xml_file.replace('.xml', '.jpg')
    if img_name in image_files:
        # copy image to output folder
        img_path = os.path.join(image_folder, img_name)
        output_img_path = os.path.join(output_images, img_name)

        # Copy the image file
        if os.path.exists(img_path):
            shutil.copy(img_path, output_img_path)

a_file = os.path.join(xml_folder, xml_files[0])

for xml_file in xml_files:
    a_file = os.path.join(xml_folder, xml_file)
    if not os.path.exists(a_file):
        continue

    img_name = xml_file.replace('.xml', '.jpg')
    if img_name not in image_files:
        continue
    
    img_path = os.path.join(image_folder, img_name)
    if not os.path.exists(img_path):
        continue
    
    root = ET.parse(a_file)
    root = root.getroot()

    # Create YOLO format annotation file
    yolo_annot_file = os.path.join(output_labels, img_name.replace('.jpg', '.txt'))
    with open(yolo_annot_file, 'w') as f:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            
            img_width = int(root.find('size').find('width').text)
            img_height = int(root.find('size').find('height').text)
            print(f"Processing {img_name} with class {class_name}")
            if class_name == 'Deer':
                bbox = obj.find('bndbox')
                x_center = (int(bbox.find('xmin').text) + int(bbox.find('xmax').text)) / 2 / img_width
                y_center = (int(bbox.find('ymin').text) + int(bbox.find('ymax').text)) / 2 / img_height
                width = (int(bbox.find('xmax').text) - int(bbox.find('xmin').text)) / img_width
                height = (int(bbox.find('ymax').text) - int(bbox.find('ymin').text)) / img_height
                f.write(f"0 {x_center} {y_center} {width} {height}\n")
                print(f"Processed {img_name} with class {class_name}: ({x_center}, {y_center}, {width}, {height})")

