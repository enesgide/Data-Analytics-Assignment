import numpy as np

def vandermonde(t, n):
    m = t.shape[0]
    A = np.zeros((m,n))

    for i in range(m):
        for j in range(n):
            A[i,j] = t[i]**j

    return A