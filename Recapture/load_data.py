from PIL import Image
import os
import numpy as np
from sklearn.utils import shuffle
from features.RID_SPIC22 import *
from tqdm import tqdm

def count_images(images_path):
    total_images = sum(len(files) for _, _, files in os.walk(images_path))
    return total_images

def load_data(images_path):
    images = []
    total_images = count_images(images_path)
    folder_name = os.path.basename(images_path)
    progress_bar = tqdm(total=total_images, desc=f"Chargement des images du dossier {folder_name}", unit="image")
    for subdir, _, files in os.walk(images_path):
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

if os.path.exists(output_file):
    replace = input("Le fichier features_and_labels.npz existe déjà. Voulez-vous le remplacer ? (o/n): ")
    if replace.lower() != 'o':
        print("Annulation de l'exécution.")
        exit()

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

np.savez(output_file, features=features, labels=labels_dataset_shuffled)
print(f"Images vectorisées et labels sauvegardés dans {output_file}")