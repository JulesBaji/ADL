from PIL import Image
import os
import numpy as np
from features.RID_SPIC22 import *

def load_data(images_path):
    images = []
    for subdir, _, files in os.walk(images_path):
        for filename in files:
            if filename.endswith((".png", ".JPG")):
                image_path = os.path.join(subdir, filename)
                image = Image.open(image_path)
                image = image.resize((600, 900))
                images.append(image)
    return images

captured_images_path = "Recapture/SingleCaptureImages"
recaptured_images_path = "Recapture/RecapturedImages"

captured_images = load_data(captured_images_path)
recaptured_images = load_data(recaptured_images_path)

captured_labels = np.zeros(len(captured_images))
recaptured_labels = np.ones(len(recaptured_images))

images_dataset = np.concatenate([captured_images, recaptured_images])
labels_dataset = np.concatenate([captured_labels, recaptured_labels])

print(images_dataset.shape, labels_dataset.shape)

features = np.array([RID_SPIC22(image) for image in images_dataset])
np.savez('features_and_labels.npz', features=features, labels=labels_dataset)
