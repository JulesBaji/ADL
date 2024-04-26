from model import *
from show import *
from generate_train_image_pairs import *
from generate_test_image_pairs import *
from RID_SPIC_model import *
from sklearn.datasets import fetch_olivetti_faces
from tensorflow.keras import backend as k
import numpy as np


def euclidean_distance(vectors):
    (featA, featB) = vectors
    sum_squared = k.sum(k.square(featA - featB), axis=1, keepdims=True)
    return k.sqrt(k.maximum(sum_squared, k.epsilon()))

# Préparation
olivetti = fetch_olivetti_faces()
images_dataset = olivetti.images # de forme (400, 64, 64), dtype=float32, valeurs entre 0.0 et 1.0
labels_dataset = olivetti.target # de forme (400,), dtype=int32, labels 

# Etape 2 : création du modèle d’extraction de features (vecteurs de descripteurs)
# feature_extractor = create_model()
# imgA = Input(shape=(64, 64, 1))
# imgB = Input(shape=(64, 64, 1))
# featA = feature_extractor(imgA)
# featB = feature_extractor(imgB)

feature_extractor = RID_SPIC_model()
imgA = Input(shape=(64, 64, 3))
imgB = Input(shape=(64, 64, 3))
featA = feature_extractor(imgA)
featB = feature_extractor(imgB)

# Etape 3 : calcul de la distance euclidienne entre deux vecteurs de descripteurs 
distance = Lambda(euclidean_distance)([featA, featB])
outputs = Dense(1, activation="sigmoid")(distance) # sigmoid : voir ci-dessous
model = Model(inputs=[imgA, imgB], outputs=outputs)

# Etape 4 : apprentissage 
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Etape 5 : prédictions
images_pair, labels_pair = generate_train_image_pairs(images_dataset, labels_dataset)
history = model.fit([images_pair[:, 0], images_pair[:, 1]], labels_pair[:],validation_split=0.1,batch_size=64,epochs=100)

image = images_dataset[92] # a random image as test image
test_image_pairs, test_label_pairs = generate_test_image_pairs(images_dataset, labels_dataset, image) # produce an array of test image pairs and test label pairs

same_face_predictions = []
different_face_predictions = []
# for each pair in the test image pair, predict the similarity between the images
for index, pair in enumerate(test_image_pairs):
    pair_image1 = np.expand_dims(pair[0], axis=-1)
    pair_image1 = np.expand_dims(pair_image1, axis=0)
    pair_image2 = np.expand_dims(pair[1], axis=-1)
    pair_image2 = np.expand_dims(pair_image2, axis=0)
    prediction = model.predict([pair_image1, pair_image2])[0][0]
    if test_label_pairs[index] == 1:
        same_face_predictions.append(prediction)
    else:
        different_face_predictions.append(prediction) 
        
print(same_face_predictions)
print(different_face_predictions)
