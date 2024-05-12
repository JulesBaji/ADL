from features.RID_SPIC22 import *
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

def load_data(images_path):
    images = []
    files_list = os.listdir(images_path)
    folder_name = os.path.basename(images_path)
    for filename in tqdm(files_list, desc=f"Chargement des images du dossier {folder_name}", unit="image"):
        if filename.lower().endswith((".png",".jpg")):
            image_path = os.path.join(images_path, filename)
            image = Image.open(image_path)
            image = image.resize((600, 900))
            images.append(image)
    return images

# Chargement des images
image_path = "Recapture/RecapturedImages/Recapture_manuelle"
images = load_data(image_path)

# Vectorisation des images
features = np.array([RID_SPIC22(np.array(image)) for image in tqdm(images, desc="Vectorisation des images", unit="image")])

# Chargement du modèle
model = load_model('detection_recapture.h5')

# Prédictions
predictions = model.predict(features)
percentage_recapture = (np.sum(predictions > 0.5) / len(predictions)) * 100
print("Prédictions réussies : ", percentage_recapture, "%")
