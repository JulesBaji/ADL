from load_data import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np

data = np.load('features_and_labels.npz')
features = data['features']
labels_dataset = data['labels']

X_train, X_val, y_train, y_val = train_test_split(features, labels_dataset, test_size=0.2, random_state=2)

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

optimizer = Adam(0.0001)
model.compile(optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    batch_size=32,
                    epochs=1000)

loss, accuracy = model.evaluate(X_val, y_val)
model.save("detection_recapture.h5")

