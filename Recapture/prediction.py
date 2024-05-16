from features.RID_SPIC22 import *
from tensorflow.keras.models import load_model
import numpy as np
import os
from sklearn.metrics import accuracy_score

# Chargement du modèle
output_dir = "modele"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# features_and_labels_dataset_only.npz pour le dataset sans le dataset manuel, features_and_labels.npz pour le dataset complet
features_file = os.path.join(output_dir, "features_and_labels_dataset_only.npz")
# features_file = os.path.join(output_dir, "features_and_labels.npz")

# detection_recapture_dataset_only.h5 pour le modèle entraîné sans le dataset manuel, detection_recapture.h5 pour le modèle entraîné avec le dataset complet
modele_file = os.path.join(output_dir, "detection_recapture_dataset_only.h5")
# modele_file = os.path.join(output_dir, "detection_recapture.h5")

model = load_model(modele_file)

data = np.load(features_file)
test_features = data['test_features']
test_labels = data['test_labels']

# Prédictions
predictions = model.predict(test_features)
predicted_labels = (predictions > 0.5).astype(int).flatten()

# Calcul de la précision
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")