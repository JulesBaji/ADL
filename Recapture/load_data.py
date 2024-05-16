from PIL import Image
import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from features.RID_SPIC22 import *
from tqdm import tqdm

def count_images(images_path, exclude_dirs=None):
    total_images = 0
    for subdir, _, files in os.walk(images_path):
        if exclude_dirs and os.path.basename(subdir) in exclude_dirs:
            continue
        total_images += len(files)
    return total_images

def load_data(images_path, exclude_dirs=None):
    images = []
    total_images = count_images(images_path, exclude_dirs)
    folder_name = os.path.basename(images_path)
    progress_bar = tqdm(total=total_images, desc=f"Chargement des images du dossier {folder_name}", unit="image")
    for subdir, _, files in os.walk(images_path):
        if exclude_dirs and os.path.basename(subdir) in exclude_dirs:
            continue
        for filename in files:
            if filename.lower().endswith((".png", ".jpg")):
                image_path = os.path.join(subdir, filename)
                image = Image.open(image_path)
                image = image.resize((600, 900))
                images.append(image)
                progress_bar.update(1)
                
    progress_bar.close()
    return images

output_dir = "modele"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_file = os.path.join(output_dir, "features_and_labels.npz")



captured_images_path = "Recapture/SingleCaptureImages"
recaptured_images_path = "Recapture/RecapturedImages"

exclude_dirs = ["Recapture_manuelle", "Capture_manuelle"]
captured_images, recaptured_images = None, None

if exclude_manual_dataset := input("Voulez-vous exclure les images des dossiers 'Recapture_manuelle' et 'Capture_manuelle' ? (o/n): ") == 'o':
    output_file = os.path.join(output_dir, "features_and_labels_dataset_only.npz")

    if os.path.exists(output_file):
        replace = input(f"Le fichier {output_file} existe déjà. Voulez-vous le remplacer ? (o/n): ")
        if replace.lower() != 'o':
            print("Annulation de l'exécution.")
            exit()

    captured_images = load_data(captured_images_path, exclude_dirs)
    recaptured_images = load_data(recaptured_images_path, exclude_dirs)
else:   
    if os.path.exists(output_file):
        replace = input(f"Le fichier {output_file} existe déjà. Voulez-vous le remplacer ? (o/n): ")
        if replace.lower() != 'o':
            print("Annulation de l'exécution.")
            exit()

    captured_images = load_data(captured_images_path)
    recaptured_images = load_data(recaptured_images_path)

captured_labels = np.zeros(len(captured_images))
recaptured_labels = np.ones(len(recaptured_images))

images_dataset = np.concatenate([captured_images, recaptured_images])
labels_dataset = np.concatenate([captured_labels, recaptured_labels])

images_dataset_shuffled, labels_dataset_shuffled = shuffle(images_dataset, labels_dataset, random_state=42)

# Split the dataset into 90% training and 10% test
train_images, test_images, train_labels, test_labels = train_test_split(
    images_dataset_shuffled, labels_dataset_shuffled, test_size=0.1, random_state=42
)

# Vectorize the training images
train_features = np.array([RID_SPIC22(image) for image in tqdm(train_images, desc="Vectorisation des images d'entraînement", unit="image")])
# Vectorize the test images
test_features = np.array([RID_SPIC22(image) for image in tqdm(test_images, desc="Vectorisation des images de test", unit="image")])

np.savez(output_file, train_features=train_features, train_labels=train_labels, test_features=test_features, test_labels=test_labels)
print(f"Images vectorisées et labels sauvegardés dans {output_file}")