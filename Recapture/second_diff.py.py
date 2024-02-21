import numpy as np

def getSecondDiff(Matrix):
    m, n, _ = Matrix.shape
    
    central = Matrix[1:m-1, 1:n-1, :]
    left_upper = Matrix[:m-2, :n-2, :]
    upper = Matrix[:m-2, 1:n-1, :]
    right_upper = Matrix[:m-2, 2:n, :]
    right = Matrix[1:m-1, 2:n, :]
    right_lower = Matrix[2:m, 2:n, :]
    lower = Matrix[2:m, 1:n-1, :]
    left_lower = Matrix[2:m, :n-2, :]
    left = Matrix[1:m-1, :n-2, :]
    
    V = 2 * central - upper - lower
    H = 2 * central - left - right
    D1 = 2 * central - left_upper - right_lower
    D2 = 2 * central - right_upper - left_lower
    
    return V, H, D1, D2
