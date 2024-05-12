from PIL import Image
import os
import numpy as np
from sklearn.utils import shuffle
from features.RID_SPIC22 import *
from tqdm import tqdm  # Importez tqdm

def load_data(images_path):
    images = []
    files_list = os.listdir(images_path)
    for filename in tqdm(files_list, desc="Chargement des images", unit="image"):
        if filename.lower().endswith((".png",".jpg")):
            image_path = os.path.join(images_path, filename)
            image = Image.open(image_path)
            image = image.resize((600, 900))
            images.append(image)
    return images

captured_images_path = "Recapture/SingleCaptureImages"
recaptured_images_path = "Recapture/RecapturedImages"

captured_images = load_data(captured_images_path)
recaptured_images = load_data(recaptured_images_path)

captured_labels = np.zeros(len(captured_images))
recaptured_labels = np.ones(len(recaptured_images))

images_dataset = np.concatenate([captured_images, recaptured_images])
labels_dataset = np.concatenate([captured_labels, recaptured_labels])

images_dataset_shuffled, labels_dataset_shuffled = shuffle(images_dataset, labels_dataset)

features = np.array([RID_SPIC22(image) for image in tqdm(images_dataset_shuffled, desc="Vectorisation des images", unit="image")])
np.savez('features_and_labels.npz', features=features, labels=labels_dataset_shuffled)
print("Features and labels saved successfully!")