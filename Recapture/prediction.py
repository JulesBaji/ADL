from features.RID_SPIC22 import *
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((600, 900))
    return image

image_path = "Recapture/test images/images.jpeg"
image = load_image(image_path)
features = np.array(RID_SPIC22(np.array(image)))

model = load_model('detection_recapture.h5')
prediction_captured = model.predict(np.expand_dims(features, axis=0))

if prediction_captured[0][0] > 0.5:
    print(prediction_captured[0][0], " : l'image est une recapture.")
else:
    print(prediction_captured[0][0], " : l'image n'est pas une recapture.")