import numpy as np

# Calcualting prediction error maps
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

def CalError(M1, M2, M3, M4, threshold):
    m, n, z = M1.shape
    ErrorMap = np.zeros((m, n, z))

    for channel in range(z):
        Q = np.stack([np.abs(M2[:, :, channel]).reshape(m * n),
                      np.abs(M3[:, :, channel]).reshape(m * n),
                      np.abs(M4[:, :, channel]).reshape(m * n)], axis=1)
        v = np.abs(M1[:, :, channel]).reshape(m * n)

        v_less_thres = v < threshold
        Q_less_thres = np.sum(Q < threshold, axis=1) > 0
        index_thres = np.logical_or(v_less_thres, Q_less_thres)

        v[index_thres] = 0
        Q[index_thres, :] = 0

        w = np.linalg.lstsq(Q.T @ Q, Q.T @ v, rcond=None)[0]

        Error = np.log2(v + np.finfo(float).eps) - np.log2(np.abs(Q @ w) + np.finfo(float).eps)
        Error = Error.reshape((m, n))
        ErrorMap[:, :, channel] = Error

    return ErrorMap