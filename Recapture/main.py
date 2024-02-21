from scipy.ndimage import imread
from skimage.color import rgb2hsv
from second_diff import getSecondDiff
from predict_error import getPredictError
from ltc_down import getLTC_down
import numpy as np

def main(image_path):
    f = imread(image_path) / 255.0

    if len(f.shape) == 2:
        print('L\'image en entrée doit être en couleur')
        features = None
    else:
        HSV = rgb2hsv(f)
        
        V, H, D1, D2 = getSecondDiff(HSV)
        threshold = 3 / 255.0
        Direction = 3
        PredictError = getPredictError(V, H, D1, D2, threshold, Direction)
        
        features_1 = np.concatenate([getLTC_down(PredictError[:, :, i]) for i in range(PredictError.shape[2])])
        
        m, n, _ = HSV.shape
        if m % 2 == 0:
            m -= 1
        if n % 2 == 0:
            n -= 1
        AvgPoolImage = (HSV[1::2, 1::2] + HSV[1::2, 2::2] + HSV[2::2, 1::2] + HSV[2::2, 2::2]) / 4.0
        
        V, H, D1, D2 = getSecondDiff(AvgPoolImage)
        PredictError = getPredictError(V, H, D1, D2, threshold, Direction)
        
        features_2 = np.concatenate([getLTC_down(PredictError[:, :, i]) for i in range(PredictError.shape[2])])
        
        features = np.concatenate((features_1, features_2))

    print(features)

if __name__ == "__main__":
    image_path = 'example_image.jpg'
    main(image_path)
