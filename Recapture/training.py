from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
import os
from tensorflow.keras.regularizers import l2

output_dir = "modele"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# features_and_labels_dataset_only.npz pour le dataset sans le dataset manuel, features_and_labels.npz pour le dataset complet
features_file = os.path.join(output_dir, "features_and_labels_dataset_only.npz")
# features_file = os.path.join(output_dir, "features_and_labels.npz")

# detection_recapture_dataset_only.h5 pour le modèle entraîné sans le dataset manuel, detection_recapture.h5 pour le modèle entraîné avec le dataset complet
modele_file = os.path.join(output_dir, "detection_recapture_dataset_only.h5")
# modele_file = os.path.join(output_dir, "detection_recapture.h5")

data = np.load(features_file)
features = data['train_features']
labels_dataset = data['train_labels']

X_train, X_val, y_train, y_val = train_test_split(features, labels_dataset, test_size=0.2, random_state=2)

if os.path.exists(modele_file):
    model = load_model(modele_file)
    print("Modèle existant chargé.")
else:
    # 2x324
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

optimizer = Adam(0.0001)
model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1000, min_lr=1e-7)

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    batch_size=32,
                    epochs=10000,
                    callbacks=reduce_lr)

loss, accuracy = model.evaluate(X_val, y_val)
model.save(features_file)
print(f"Modèle enregistré dans {features_file}")