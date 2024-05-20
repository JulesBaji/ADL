from features.RID_SPIC22 import *
from tensorflow.keras.models import load_model
import numpy as np
import os
from sklearn.metrics import accuracy_score

# Création du répertoire de sortie pour enregistrer les modèles et les fichiers de caractéristiques s'il n'existe pas déjà
output_dir = "modele"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Définition du chemin vers le fichier contenant les caractéristiques et les labels de test
# Utiliser 'features_and_labels_dataset_only.npz' pour le dataset sans les images manuelles,
# et 'features_and_labels.npz' pour le dataset complet
# features_file = os.path.join(output_dir, "features_and_labels_dataset_only.npz")
features_file = os.path.join(output_dir, "features_and_labels.npz")

# Définition du chemin vers le fichier de modèle
# Utiliser 'detection_recapture_dataset_only.h5' pour le modèle entraîné sans les images manuelles,
# et 'detection_recapture.h5' pour le modèle entraîné avec le dataset complet
# modele_file = os.path.join(output_dir, "detection_recapture_dataset_only.h5")
modele_file = os.path.join(output_dir, "detection_recapture.h5")

# Chargement du modèle Keras depuis le fichier spécifié
model = load_model(modele_file)

# Chargement des données de test à partir du fichier .npz
data = np.load(features_file)
test_features = data['test_features']
test_labels = data['test_labels']

# Prédictions sur les caractéristiques de test en utilisant le modèle chargé
predictions = model.predict(test_features)
# Conversion des prédictions en étiquettes binaires (0 ou 1) en utilisant un seuil de 0.5
predicted_labels = (predictions > 0.5).astype(int).flatten()

# Calcul de la précision en comparant les étiquettes de test réelles avec les étiquettes prédites
accuracy = accuracy_score(test_labels, predicted_labels)
# Affichage de la précision en pourcentage
print(f"Accuracy: {accuracy * 100:.2f}%")