import numpy as np
from skimage import color
# from features.second_diff import *
# from features.predict_error import *
# from features.ltc_down import *
from second_diff import *
from predict_error import *
from ltc_down import *

 # image: a RGB image's name
def RID_SPIC22(image):
       
    threshold = 3 / 255
    Direction = 3   
    f = image / 255.0
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

    # AvgPoolImage = np.zeros((m // 2, n // 2, 3))
    # m_pool, n_pool, _ = AvgPoolImage.shape
    # for k in range(3):
    #     AvgPoolImage[:, :, k] = (HSV[:m_pool*2:2, :n_pool*2:2, k] + HSV[1:m_pool*2:2, :n_pool*2:2, k] + HSV[:m_pool*2:2, 1:n_pool*2:2, k] + HSV[1:m_pool*2:2, 1:n_pool*2:2, k]) / 4

    AvgPoolImage = np.zeros((m // 2, n // 2, 3))
    for k in range(3):
        AvgPoolImage[:,:,k] = (HSV[1:m:2, 1:n:2, k] + HSV[1:m:2, 2:n:2, k] + HSV[2:m:2, 1:n:2, k] + HSV[2:m:2, 2:n:2, k]) / 4

    
    V, H, D1, D2 = getSecondDiff(AvgPoolImage)  
    PredictError = getPredictError(V, H, D1, D2, threshold, Direction)
    
    features_2 = []
    for i in range(PredictError.shape[2]):
        features_2.extend(getLTC_down(PredictError[:, :, i]))
    
    features = np.concatenate((features_1, features_2))
    return features

from PIL import Image
image_path = 'Recapture/SingleCaptureImages/D40/DS-05-0015-S%D40.JPG'
image = Image.open(image_path)
print(RID_SPIC22(np.array(image)))