from PIL import Image
import os
import re
import numpy as np

def extract_label(filename):
    if filename.endswith(".JPG"):
        match = re.search(r'-(\d+)-S%D70S\.(?:jpg|JPG)$', filename)
    else:
        match = re.search(r'-(\d+)\.(?:png|PNG)$', filename)
    if match:
        return match.group(1).lstrip("0")
    else:
        return None

def load_data(images_path):
    images = []
    labels = []
    for filename in os.listdir(images_path):
        if filename.endswith((".png", ".JPG")):
            image_path = os.path.join(images_path, filename)
            image = Image.open(image_path)
            image = image.resize((600, 900))
            images.append(image)
            label = extract_label(filename)
            labels.append(label)
    return images, labels

def filter_images_and_labels(captured_images, captured_labels, recaptured_images, recaptured_labels):
    filtered_captured_images = []
    filtered_recaptured_images = []
    for image, label in zip(captured_images, captured_labels):
        if label in recaptured_labels:
            filtered_captured_images.append(image)
            filtered_recaptured_images.append(recaptured_images[recaptured_labels.index(label)])
    
    images_dataset = np.concatenate([filtered_captured_images, filtered_recaptured_images])
    filtered_captured_labels = np.zeros(len(filtered_captured_images))
    filtered_recaptured_labels = np.ones(len(filtered_recaptured_images))
    labels_dataset = np.concatenate([filtered_captured_labels, filtered_recaptured_labels])
    return images_dataset, labels_dataset



