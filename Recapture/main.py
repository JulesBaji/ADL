import numpy as np
from skimage import io, color
from second_diff import *
from predict_error import *
from ltc_down import *

 # image: a RGB image's name
def main(image_path):
    
    threshold = 3 / 255
    Direction = 3   
    f = (io.imread(image_path) / 255.0)
    m, n, z = f.shape

    if z == 1:
        print('The input image must be color image')
        return []
    
    HSV = color.rgb2hsv(f)
    V, H, D1, D2 = getSecondDiff(HSV) # Second order difference
    PredictError = getPredictError(V, H, D1, D2, threshold, Direction)
    
    features_1 = []
    for i in range(PredictError.shape[2]):
        features_1.extend(getLTC_down(PredictError[:, :, i]))
    
    #calculate average pooled image 
    if m % 2 == 0:
        m -= 1
    if n % 2 == 0:
        n -= 1
        
    AvgPoolImage = np.zeros((m // 2, n // 2, 3))
    m_pool, n_pool, _ = AvgPoolImage.shape
    for k in range(3):
        AvgPoolImage[:, :, k] = (HSV[:m_pool*2:2, :n_pool*2:2, k] + HSV[1:m_pool*2:2, :n_pool*2:2, k] + HSV[:m_pool*2:2, 1:n_pool*2:2, k] + HSV[1:m_pool*2:2, 1:n_pool*2:2, k]) / 4

    
    V, H, D1, D2 = getSecondDiff(AvgPoolImage)  
    PredictError = getPredictError(V, H, D1, D2, threshold, Direction)
    
    features_2 = []
    for i in range(PredictError.shape[2]):
        features_2.extend(getLTC_down(PredictError[:, :, i]))
    
    features = np.concatenate((features_1, features_2))
    return features
    
if __name__ == "__main__":
    image_path = 'test32x21.jpg'
    main(image_path)