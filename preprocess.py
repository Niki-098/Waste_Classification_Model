import os
import cv2
import numpy as np
import random
import shutil

# class folders and mappings
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
class_to_id = {cls: idx for idx, cls in enumerate(classes)}

#Root dir with class folders 
root_dir = 'data'  

#Output YOLO dataset dir 
output_dir = 'Dataset'  
os.makedirs(os.path.join(output_dir, 'train', 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'train', 'labels'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val', 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val', 'labels'), exist_ok=True)

#Optional: Preview dir for visualized bboxes
preview_dir = os.path.join(output_dir, 'previews')
os.makedirs(preview_dir, exist_ok=True)

#Function to auto-generate bbox using thresholding + contours (assumes uniform background)
def generate_bbox(image_path, threshold=240): 
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Largest contour as the object
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Normalize (0-1)
    img_h, img_w = img.shape[:2]
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    width = w / img_w
    height = h / img_h

    return x_center, y_center, width, height, (x, y, w, h)  # Return normalized + denormalized for drawing

#Optional: Visualize bbox on image and save preview
def visualize_bbox(image_path, bbox_denorm, output_path, label):
    if bbox_denorm:
        img = cv2.imread(image_path)
        x, y, w, h = bbox_denorm
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imwrite(output_path, img)

#Set to True to generate preview images for verification
visualize = True 

#Total image counter
total_images = 0

#Process each class
for cls in classes:
    class_dir = os.path.join(root_dir, cls)
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    total_images += len(images)

    #Shuffle for random split
    random.shuffle(images)

    # Split: 8:2
    split_idx = int(0.8 * len(images))
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Process train
    for img_name in train_images:
        src_path = os.path.join(class_dir, img_name)
        dst_img_path = os.path.join(output_dir, 'train', 'images', img_name)
        shutil.copy(src_path, dst_img_path)

        result = generate_bbox(src_path)
        if result:
            x_center, y_center, width, height, bbox_denorm = result
            label_path = os.path.join(output_dir, 'train', 'labels', img_name.rsplit('.', 1)[0] + '.txt')
            with open(label_path, 'w') as f:
                f.write(f"{class_to_id[cls]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            if visualize:
                preview_path = os.path.join(preview_dir, f"train_{img_name}")
                visualize_bbox(src_path, bbox_denorm, preview_path, cls)

    # Process val
    for img_name in val_images:
        src_path = os.path.join(class_dir, img_name)
        dst_img_path = os.path.join(output_dir, 'val', 'images', img_name)
        shutil.copy(src_path, dst_img_path)

        result = generate_bbox(src_path)
        if result:
            x_center, y_center, width, height, bbox_denorm = result
            label_path = os.path.join(output_dir, 'val', 'labels', img_name.rsplit('.', 1)[0] + '.txt')
            with open(label_path, 'w') as f:
                f.write(f"{class_to_id[cls]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            if visualize:
                preview_path = os.path.join(preview_dir, f"val_{img_name}")
                visualize_bbox(src_path, bbox_denorm, preview_path, cls)

print(f"Dataset prepared with {total_images} total images. Train: ~{int(0.8 * total_images)}, Val: ~{int(0.2 * total_images)}.")
if visualize:
    print(f"Preview images with drawn bboxes saved in: {preview_dir}. Check them to verify accuracy.")
print("Update data.yaml path to:", output_dir)