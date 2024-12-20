import numpy as np
from numpy import typing as npt
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF, _singularize

# Insert your package here


"""
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix

    HINTS:
    (1) Normalize the input pts1 and pts2 using the matrix T.
    (2) Setup the eight point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Use the function `_singularize` (provided) to enforce the singularity condition. 
    (5) Use the function `refineF` (provided) to refine the computed fundamental matrix. 
        (Remember to use the normalized points instead of the original points)
    (6) Unscale the fundamental matrix
"""


def eightpoint(pts1: npt.NDArray, pts2: npt.NDArray, M):
    # Replace pass by your implementation
    # ----- TODO -----
    # YOUR CODE HERE

    num_points = pts1.shape[0]

    pts1_homogenous = np.ones((num_points, 3)) # N x 3
    pts2_homogenous = np.ones((num_points, 3)) # N x 3

    pts1_homogenous[:, :2] = pts1
    pts2_homogenous[:, :2] = pts2

    # Create a Matrix T such that it normalizes the points (Check normalization implementation twice)
    T = np.array([[1/M, 0, 0],
                  [0, 1/M, 0],
                  [0, 0, 1]])
    
    pts1_normalized = (T @ pts1_homogenous.T).T # N x 3
    pts2_normalized = (T @ pts2_homogenous.T).T # N x 3

    # Setup the fundamental matrix equation given N points    
    coeffeciant_matrix = np.zeros((num_points, 9)) # N x 9

    for i in range(0, num_points):

        u1 = pts1_normalized[i][0]; u2 = pts2_normalized[i][0] #u1 and u2 are x coordinate pixel values for left and right images respectively
        v1 = pts1_normalized[i][1]; v2 = pts2_normalized[i][1] #v1 and v2 are y coordinate pixel values for left and right images respectively

        coeffeciant_vector = np.array([u1*u2, u1*v2, u1, v1*u2, v1*v2, v1, u2, v2, 1])
        coeffeciant_matrix[i, :] = coeffeciant_vector

    # Use SVD to find the fundamental matrix
    u, s, vh = np.linalg.svd(coeffeciant_matrix)
    f_vector = vh[-1]
    f_matrix = f_vector.reshape((3,3)).T

    # Use _singularize to ensure matrix rank is 2 for non trivial solution
    # This enforces the epipolar lines to snap together on the epipole
    f_matrix = _singularize(f_matrix)

    # Use refineF to refine the computed fundamental matrix using normalized points instead of original points
    f_matrix = refineF(f_matrix, pts1_normalized[:, :2], pts2_normalized[:, :2])

    # Unscale the fundamental matrix
    f_un = T.T @ f_matrix @ T
    f_un /= f_un[-1, -1]
    return f_un


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    M = np.max([*im1.shape, *im2.shape])
    np.savez("q2_1.npz", F, M)
    print(F)

    # Q2.1
    displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)
    print(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)))

    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2
    assert np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1
