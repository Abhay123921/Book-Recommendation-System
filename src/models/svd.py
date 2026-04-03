from scipy.sparse.linalg import svds
import numpy as np

def train_svd(matrix, k=100):

    # Convert to numpy
    matrix_values = matrix.values

    # -------------------------------
    # Step 1: Compute global mean
    # -------------------------------
    mean = np.mean(matrix_values)

    # -------------------------------
    # Step 2: Mean center
    # -------------------------------
    matrix_centered = matrix_values - mean

    # -------------------------------
    # Step 3: SVD
    # -------------------------------
    U, sigma, Vt = svds(matrix_centered, k=k)
    sigma = np.diag(sigma)

    # -------------------------------
    # Step 4: Reconstruct
    # -------------------------------
    predicted = np.dot(np.dot(U, sigma), Vt) 

    return predicted
