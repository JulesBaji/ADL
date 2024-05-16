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

data_file = os.path.join(output_dir, 'features_and_labels.npz')
data = np.load(data_file)
features = data['train_features']
labels_dataset = data['train_labels']

X_train, X_val, y_train, y_val = train_test_split(features, labels_dataset, test_size=0.2, random_state=2)

output_file = os.path.join(output_dir, "detection_recapture.h5")
if os.path.exists(output_file):
    model = load_model(output_file)
    print("Modèle existant chargé.")
else:
    # 2x324
    model = Sequential([
        Dense(324, activation='relu', input_shape=(324,), kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    print("Nouveau modèle créé.")

optimizer = Adam(0.0001)
model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=1000, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=500, min_lr=1e-6)

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    batch_size=32,
                    epochs=1000,
                    callbacks=[early_stopping, reduce_lr])

loss, accuracy = model.evaluate(X_val, y_val)
model.save(output_file)
print(f"Modèle enregistré dans {output_file}")