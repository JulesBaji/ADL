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

# Diviser les données en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(features, labels_dataset, test_size=0.1, random_state=2)

# Définir le modèle
model = Sequential([
    Dense(128, activation='relu', input_shape=(324,)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compiler le modèle
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entraîner le modèle
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    batch_size=2,
                    epochs=100)

# Évaluer le modèle
loss, accuracy = model.evaluate(X_val, y_val)
print("Loss:", loss)
print("Accuracy:", accuracy)

# # Etape 2 : création du modèle d’extraction de features (vecteurs de descripteurs)
# feature_extractor = RID_SPIC_model()
# imgA = Input(shape=(900, 600, 3))
# imgB = Input(shape=(900, 600, 3))
# featA = feature_extractor(imgA)
# featB = feature_extractor(imgB)

# # Etape 3 : calcul de la distance euclidienne entre deux vecteurs de descripteurs 
# distance = Lambda(euclidean_distance)([featA, featB])
# outputs = Dense(1, activation="sigmoid")(distance) # sigmoid : voir ci-dessous
# model = Model(inputs=[imgA, imgB], outputs=outputs)

# # Etape 4 : apprentissage 
# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# images_pair, labels_pair = generate_train_image_pairs(images_dataset, labels_dataset)
# history = model.fit([images_pair[:, 0], images_pair[:, 1]], labels_pair[:],validation_split=0.1,batch_size=64,epochs=100)

# # Etape 5 : prédictions
# image = images_dataset[0] # a random image as test image
# test_image_pairs, test_label_pairs = generate_test_image_pairs(images_dataset, labels_dataset, image) # produce an array of test image pairs and test label pairs

# same_face_predictions = []
# different_face_predictions = []
# # for each pair in the test image pair, predict the similarity between the images
# for index, pair in enumerate(test_image_pairs):
#     pair_image1 = np.expand_dims(pair[0], axis=-1)
#     pair_image1 = np.expand_dims(pair_image1, axis=0)
#     pair_image2 = np.expand_dims(pair[1], axis=-1)
#     pair_image2 = np.expand_dims(pair_image2, axis=0)
#     prediction = model.predict([pair_image1, pair_image2])[0][0]
#     if test_label_pairs[index] == 1:
#         same_face_predictions.append(prediction)
#     else:
#         different_face_predictions.append(prediction) 
        
# print(same_face_predictions)
# print(different_face_predictions)
