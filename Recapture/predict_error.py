import numpy as np

def CalError(M1, M2, M3, M4, threshold):
    m, n, z = M1.shape
    ErrorMap = []
    
    for channel in range(z):
        Q = np.array([M2[:,:,channel].flatten(), M3[:,:,channel].flatten(), M4[:,:,channel].flatten()]).T
        v = M1[:,:,channel].flatten()
        
        v_less_thres = (v < threshold)
        Q_less_thres = (np.sum((Q < threshold), axis=1) > 0)
        index_thres = (v_less_thres + Q_less_thres) > 0
        v[index_thres] = 0
        Q[index_thres, :] = 0
        
        w = np.linalg.lstsq(Q.T.dot(Q), Q.T.dot(v), rcond=None)[0]
        
        Error = np.log2(v + np.finfo(float).eps) - np.log2(np.abs(Q.dot(w)) + np.finfo(float).eps)
        Error = Error.reshape((m, n))
        
        ErrorMap.append(Error)
    
    return np.array(ErrorMap)

def getPredictError(V, H, D1, D2, threshold, Direction):
    # V, H, D1, D2: Output of getSecondDiff
    # threshold: used for noise removal
    # Direction: the number of directions for calculating prediction error maps
    if Direction == 1:
        PredictError = CalError(V, H, D1, D2, threshold)
    elif Direction == 2:
        PredictError1 = CalError(V, H, D1, D2, threshold)
        PredictError2 = CalError(H, V, D1, D2, threshold)
        PredictError = np.dstack((PredictError1, PredictError2))
    elif Direction == 3:
        PredictError1 = CalError(V, H, D1, D2, threshold)
        PredictError2 = CalError(H, V, D1, D2, threshold)
        PredictError3 = CalError(D1, V, H, D2, threshold)
        PredictError = np.dstack((PredictError1, PredictError2, PredictError3))
    elif Direction == 4:
        PredictError1 = CalError(V, H, D1, D2, threshold)
        PredictError2 = CalError(H, V, D1, D2, threshold)
        PredictError3 = CalError(D1, V, H, D2, threshold)
        PredictError4 = CalError(D2, V, H, D1, threshold)
        PredictError = np.dstack((PredictError1, PredictError2, PredictError3, PredictError4))
    
    return PredictError
