import os
import cv2
import numpy as np
from tqdm import tqdm

def crop_brain_contour(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # Threshold to get binary image
    _, thresh = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY)
    # Fill holes
    kernel = np.ones((5,5), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Find contours
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return img  # fallback: return original
    # Get the largest contour
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    # Crop and return
    cropped = img[y:y+h, x:x+w]
    return cropped

def preprocess_and_save(input_root, output_root, target_size=(224,224)):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    for split in ['TRAIN', 'VAL', 'TEST']:
        for label in ['yes', 'no']:
            in_dir = os.path.join(input_root, split, label)
            out_dir = os.path.join(output_root, split, label)
            os.makedirs(out_dir, exist_ok=True)
            images = os.listdir(in_dir)
            for img_name in tqdm(images, desc=f"{split}/{label}"):
                in_path = os.path.join(in_dir, img_name)
                out_path = os.path.join(out_dir, img_name)
                try:
                    img = cv2.imread(in_path)
                    if img is None: continue
                    cropped = crop_brain_contour(img)
                    # Resize to target size after cropping
                    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
                    cv2.imwrite(out_path, resized)
                except Exception as e:
                    print(f"Failed: {in_path} ({e})")

# Usage
input_data_dir = 'brain_tumor_dataset'
output_data_dir = 'PREPROCESSED_DATA'
preprocess_and_save(input_data_dir, output_data_dir, target_size=(224,224))