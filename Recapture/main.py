from model import *
from show import *
from load_data import *
from generate_train_image_pairs import *
from generate_test_image_pairs import *
from RID_SPIC_model import *
from tensorflow.keras import backend as k
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

def euclidean_distance(vectors):
    (featA, featB) = vectors
    sum_squared = k.sum(k.square(featA - featB), axis=1, keepdims=True)
    return k.sqrt(k.maximum(sum_squared, k.epsilon()))

captured_images_path = "Recapture/SingleCaptureImages/D70S"
recaptured_images_path = "Recapture/RecapturedImages/D70s"

captured_images, captured_labels = load_data(captured_images_path)
recaptured_images, recaptured_labels = load_data(recaptured_images_path)

images_dataset, labels_dataset = filter_images_and_labels(captured_images, captured_labels, recaptured_images, recaptured_labels)

features = np.array([RID_SPIC22(image) for image in images_dataset])

X_train, X_val, y_train, y_val = train_test_split(features, labels_dataset, test_size=0.1, random_state=2)

# 2x324
model = Sequential([
    Dense(324, activation='relu', input_shape=(324,)),
    # Dropout(0.5),
    Dense(324, activation='relu', input_shape=(324,)),
    Dense(128, activation='relu', input_shape=(324,)),
    Dense(64, activation='relu'),
    # Dropout(0.5),
    Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(0.0001)
model.compile(optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    batch_size=2,
                    epochs=1000)

loss, accuracy = model.evaluate(X_val, y_val)
print("Loss:", loss)
print("Accuracy:", accuracy)
