from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import os

output_dir = "modele"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

data_file = os.path.join(output_dir, 'features_and_labels.npz')
data = np.load(data_file)
features = data['features']
labels_dataset = data['labels']

X_train, X_val, y_train, y_val = train_test_split(features, labels_dataset, test_size=0.2, random_state=2)

output_file = os.path.join(output_dir, "detection_recapture.h5")
if os.path.exists(output_file):
    model = load_model(output_file)
    print("Modèle existant chargé.")

else :
    # 2x324
    model = Sequential([
        Dense(324, activation='relu', input_shape=(324,)),
        Dropout(0.5),
        Dense(324, activation='relu', input_shape=(324,)),
        Dropout(0.5),
        Dense(128, activation='relu', input_shape=(324,)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    print("Nouveau modèle créé.")

optimizer = Adam(0.0001)
model.compile(optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    batch_size=32,
                    epochs=1000)

loss, accuracy = model.evaluate(X_val, y_val)
    
model.save(output_file)
print(f"Modèle enregistré dans {output_file}")