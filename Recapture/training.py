from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Création du répertoire de sortie pour enregistrer les modèles et les fichiers de caractéristiques s'il n'existe pas déjà
output_dir = "modele"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Définition du chemin vers le fichier contenant les caractéristiques et les labels
# Utiliser 'features_and_labels_dataset_only.npz' pour le dataset sans les images manuelles,
# et 'features_and_labels.npz' pour le dataset complet
# features_file = os.path.join(output_dir, "features_and_labels_dataset_only.npz")
features_file = os.path.join(output_dir, "features_and_labels.npz")

# Définition du chemin vers le fichier de modèle
# Utiliser 'detection_recapture_dataset_only.h5' pour le modèle entraîné sans les images manuelles,
# et 'detection_recapture.h5' pour le modèle entraîné avec le dataset complet
# modele_file = os.path.join(output_dir, "detection_recapture_dataset_only.h5")
modele_file = os.path.join(output_dir, "detection_recapture.h5")

# Chargement des données de caractéristiques et des labels à partir du fichier .npz
data = np.load(features_file)
features = data['train_features']
labels_dataset = data['train_labels']

# Séparation des données en ensembles d'entraînement et de validation (80% entraînement, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(features, labels_dataset, test_size=0.2, random_state=2)

# Chargement du modèle existant si le fichier de modèle existe, sinon création d'un nouveau modèle
if os.path.exists(modele_file):
    model = load_model(modele_file)
    print("Modèle existant chargé.")
else:
    # Création d'un nouveau modèle avec une architecture de réseau de neurones
    model = Sequential([
        Dense(324, activation='relu', input_shape=(324,)),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    print("Nouveau modèle créé.")

# Configuration de l'optimiseur et compilation du modèle avec la perte 'binary_crossentropy' et la métrique 'accuracy'
optimizer = Adam(0.0001)
model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Définition des callbacks
# Réduction du taux d'apprentissage si la perte de validation n'améliore pas pendant 500 epochs
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=500, min_lr=1e-7)

# Enregistrement du meilleur modèle basé sur la précision de validation
checkpoint_callback = ModelCheckpoint(
    filepath=modele_file,
    save_best_only=True,
    monitor='val_accuracy',  # Métrique à surveiller pour déterminer le meilleur modèle
    mode='max',  # Maximise la métrique de validation
    verbose=1
)

# Entraînement du modèle avec les données d'entraînement, en utilisant les données de validation pour l'évaluation
# Utilisation d'une taille de batch de 32 et un maximum de 20000 epochs, avec les callbacks définis précédemment
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    batch_size=32,
                    epochs=20000,
                    callbacks=[reduce_lr, checkpoint_callback])

# Évaluation du modèle sur les données de validation et affichage de la perte et de la précision
loss, accuracy = model.evaluate(X_val, y_val)
model.save(modele_file)
print(f"Modèle enregistré dans {modele_file}")