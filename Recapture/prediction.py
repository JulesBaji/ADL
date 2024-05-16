from features.RID_SPIC22 import *
from tensorflow.keras.models import load_model
import numpy as np
import os
from sklearn.metrics import accuracy_score

# Chargement du modèle
output_dir = "modele"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file_modele = os.path.join(output_dir, "detection_recapture.h5")
output_file_features = os.path.join(output_dir, "features_and_labels.npz")

model = load_model(output_file_modele)

data = np.load(output_file_features)
test_features = data['test_features']
test_labels = data['test_labels']

# Prédictions
predictions = model.predict(test_features)
predicted_labels = (predictions > 0.5).astype(int).flatten()

# Calcul de la précision
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")