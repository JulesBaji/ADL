from PIL import Image
import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from features.RID_SPIC22 import *
from tqdm import tqdm
from random import randint

# Fonction pour compter le nombre total d'images dans un répertoire, en excluant éventuellement certains sous-répertoires
def count_images(images_path, exclude_dirs=None):
    total_images = 0
    for subdir, _, files in os.walk(images_path):
        if exclude_dirs and os.path.basename(subdir) in exclude_dirs:
            continue
        total_images += len(files)
    return total_images

# Fonction pour réaliser un recadrage aléatoire suivi d'un redimensionnement d'une image
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

# Fonction pour charger les images depuis un répertoire, avec une option pour exclure certains sous-répertoires
def load_data(images_path, exclude_dirs=None):
    images = []
    total_images = count_images(images_path, exclude_dirs)
    folder_name = os.path.basename(images_path)

    # Chargement et traitement des images avec une barre de progression
    progress_bar = tqdm(total=total_images, desc=f"Chargement des images du dossier {folder_name}", unit="image")
    for subdir, _, files in os.walk(images_path):
        if exclude_dirs and os.path.basename(subdir) in exclude_dirs:
            continue
        for filename in files:
            if filename.lower().endswith((".png", ".jpg")):
                image_path = os.path.join(subdir, filename)
                image = Image.open(image_path)
                image = random_crop_and_resize(image)
                images.append(image)
                progress_bar.update(1)
                
    progress_bar.close()
    return images

# Définition du répertoire de sortie pour les fichiers de modèle et de données
output_dir = "modele"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_file = os.path.join(output_dir, "features_and_labels.npz")

captured_images_path = "Recapture/SingleCaptureImages"
recaptured_images_path = "Recapture/RecapturedImages"

# Sous-répertoires à exclure si l'utilisateur le souhaite
exclude_dirs = ["Recapture_manuelle", "Capture_manuelle"]
captured_images, recaptured_images = None, None

# Demande à l'utilisateur s'il souhaite exclure les images manuelles
if exclude_manual_dataset := input("Voulez-vous exclure les images des dossiers 'Recapture_manuelle' et 'Capture_manuelle' ? (o/n): ") == 'o':
    output_file = os.path.join(output_dir, "features_and_labels_dataset_only.npz")

    if os.path.exists(output_file):
        replace = input(f"Le fichier {output_file} existe déjà. Voulez-vous le remplacer ? (o/n): ")
        if replace.lower() != 'o':
            print("Annulation de l'exécution.")
            exit()

    # Chargement des images en excluant les répertoires spécifiés
    captured_images = load_data(captured_images_path, exclude_dirs)
    recaptured_images = load_data(recaptured_images_path, exclude_dirs)
else:   
    if os.path.exists(output_file):
        replace = input(f"Le fichier {output_file} existe déjà. Voulez-vous le remplacer ? (o/n): ")
        if replace.lower() != 'o':
            print("Annulation de l'exécution.")
            exit()

    # Chargement des images sans exclure de répertoires
    captured_images = load_data(captured_images_path)
    recaptured_images = load_data(recaptured_images_path)

# Création des labels pour les images capturées (0) et recapturées (1)
captured_labels = np.zeros(len(captured_images))
recaptured_labels = np.ones(len(recaptured_images))

# Concatenation des datasets d'images et des labels correspondants
images_dataset = np.concatenate([captured_images, recaptured_images])
labels_dataset = np.concatenate([captured_labels, recaptured_labels])

# Mélange des datasets d'images et de labels
images_dataset_shuffled, labels_dataset_shuffled = shuffle(images_dataset, labels_dataset, random_state=42)

# Division du dataset en ensembles d'entraînement (90%) et de test (10%)
train_images, test_images, train_labels, test_labels = train_test_split(
    images_dataset_shuffled, labels_dataset_shuffled, test_size=0.1, random_state=42
)

# Vectorisation des images d'entraînement
train_features = np.array([RID_SPIC22(image) for image in tqdm(train_images, desc="Vectorisation des images d'entraînement", unit="image")])
# Vectorisation des images de test
test_features = np.array([RID_SPIC22(image) for image in tqdm(test_images, desc="Vectorisation des images de test", unit="image")])

# Sauvegarde des caractéristiques vectorisées et des labels dans un fichier .npz
np.savez(output_file, train_features=train_features, train_labels=train_labels, test_features=test_features, test_labels=test_labels)
print(f"Images vectorisées et labels sauvegardés dans {output_file}")