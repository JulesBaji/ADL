from features.RID_SPIC22 import *
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from random import randint

# Définition d'une fonction pour effectuer un recadrage aléatoire et redimensionner les images
def random_crop_and_resize(image, target_size=(640, 480), aspect_ratio=(4, 3)):
    aspect_ratio_width, aspect_ratio_height = aspect_ratio
    
    original_width, original_height = image.size

    # Calcul des dimensions de recadrage en fonction du rapport d'aspect
    if original_width / aspect_ratio_width < original_height / aspect_ratio_height:
        crop_width = original_width
        crop_height = int(original_width * aspect_ratio_height / aspect_ratio_width)
    else:
        crop_height = original_height
        crop_width = int(original_height * aspect_ratio_width / aspect_ratio_height)

    # Détermination des coordonnées du recadrage aléatoire
    left = randint(0, original_width - crop_width)
    top = randint(0, original_height - crop_height)
    right = left + crop_width
    bottom = top + crop_height

    # Recadrage et redimensionnement de l'image
    image_cropped = image.crop((left, top, right, bottom))
    image_resized = image_cropped.resize(target_size)

    return image_resized

# Définition d'une fonction pour charger et prétraiter les images d'un répertoire
def load_data(images_path):
    images = []
    files_list = os.listdir(images_path)
    folder_name = os.path.basename(images_path)

    # Chargement et traitement des images avec une barre de progression
    for filename in tqdm(files_list, desc=f"Chargement des images du dossier {folder_name}", unit="image"):
        if filename.lower().endswith((".png",".jpg")):
            image_path = os.path.join(images_path, filename)
            image = Image.open(image_path)
            image = random_crop_and_resize(image)
            images.append(image)
    return images

# Chargement des images du répertoire spécifié
# image_path = "Recapture/RecapturedImages/Recapture_manuelle"
# image_path = "Recapture/SingleCaptureImages/Capture_manuelle"
# image_path = "Recapture/Capture_manuelle_tests"
image_path = "Recapture/Recapture_manuelle_tests"
images = load_data(image_path)

# Vectorisation des images en utilisant la fonction RID_SPIC22
test_features = np.array([RID_SPIC22(np.array(image)) for image in tqdm(images, desc="Vectorisation des images", unit="image")])

# Création du répertoire de sortie pour enregistrer les modèles et les fichiers de caractéristiques s'il n'existe pas déjà
output_dir = "modele"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Définition du chemin vers le fichier de modèle
# Utiliser 'detection_recapture_dataset_only.h5' pour le modèle entraîné sans les images manuelles,
# et 'detection_recapture.h5' pour le modèle entraîné avec le dataset complet
# modele_file = os.path.join(output_dir, "detection_recapture_dataset_only.h5")
modele_file = os.path.join(output_dir, "detection_recapture.h5")

# Chargement du modèle Keras depuis le fichier spécifié
model = load_model(modele_file)

# Prédictions sur les caractéristiques de test en utilisant le modèle chargé
predictions = model.predict(test_features)
# Conversion des prédictions en étiquettes binaires (0 ou 1) en utilisant un seuil de 0.5
# True == 1, False == 0 !
predicted_capture_labels = (predictions > 0.5).astype(int).flatten()

# Création de labels de test pour les images capturées (0 pour toutes les images)
test_capture_labels = np.zeros_like(predicted_capture_labels)
# Affichage de la précision pour les images capturées et recapturées
accuracy_capture = accuracy_score(test_capture_labels, predicted_capture_labels)
print(f"Pourcentage d'images capturées trouvées : {accuracy_capture * 100:.2f}%")
print(f"Pourcentage d'images recapturées trouvées : {(1 - accuracy_capture) * 100:.2f}%")