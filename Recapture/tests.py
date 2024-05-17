from features.RID_SPIC22 import *
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score

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
test_features = np.array([RID_SPIC22(np.array(image)) for image in tqdm(images, desc="Vectorisation des images", unit="image")])

# Chargement du modèle
output_dir = "modele"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# features_and_labels_dataset_only.npz pour le dataset sans le dataset manuel, features_and_labels.npz pour le dataset complet
# features_file = os.path.join(output_dir, "features_and_labels_dataset_only.npz")
features_file = os.path.join(output_dir, "features_and_labels.npz")

# detection_recapture_dataset_only.h5 pour le modèle entraîné sans le dataset manuel, detection_recapture.h5 pour le modèle entraîné avec le dataset complet
# modele_file = os.path.join(output_dir, "detection_recapture_dataset_only.h5")
modele_file = os.path.join(output_dir, "detection_recapture.h5")

model = load_model(modele_file)

# Prédictions
predictions = model.predict(test_features)
predicted_capture_labels = (predictions <= 0.5).astype(int).flatten()

# Calcul de la précision
test_capture_labels = np.zeros_like(predicted_capture_labels)
accuracy_capture = accuracy_score(test_capture_labels, predicted_capture_labels)
print(f"Pourcentage d'images capturées trouvées : {accuracy_capture * 100:.2f}%")
print(f"Pourcentage d'images recapturées trouvées : {(1 - accuracy_capture) * 100:.2f}%")