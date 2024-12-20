import numpy as np
import matplotlib.pyplot as plt

from helper import camera2
from q2_1_eightpoint import eightpoint
from q3_1_essential_matrix import essentialMatrix
from q2_2_sevenpoint import sevenpoint

# Insert your package here


"""
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.

    Hints:
    (1) For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2
    (2) Solve for the least square solution using np.linalg.svd
    (3) Calculate the reprojection error using the calculated 3D points and C1 & C2 (do not forget to convert from 
        homogeneous coordinates to non-homogeneous ones)
    (4) Keep track of the 3D points and projection error, and continue to next point 
    (5) You do not need to follow the exact procedure above. 
"""


def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    # ----- TODO -----
    # YOUR CODE HERE

    # Pts of the shape N x 2 (x, y)
    num_points = pts1.shape[0]

    # All ps are of the shape (1 x 4), equivalent to pl1.T, pl2.T ...
    pl1 = C1[0]; pl2 = C1[1]; pl3 = C1[2]
    pr1 = C2[0]; pr2 = C2[1]; pr3 = C2[2]

    w = np.zeros((num_points, 4))

    for i in range(num_points):
        # Extract relevant x and y points from pts1 and 2
        xl = pts1[i][0]; xr = pts2[i][0]
        yl = pts1[i][1]; yr = pts2[i][1]

        A_rows = np.array([yl*pl3 - pl2,
                           pl1 - xl*pl3,
                           yr*pr3 - pr2,
                           pr1 - xr*pr3])
        
        # Run SVD to get back w = [X, Y, Z, 1].T
        u, s, vh = np.linalg.svd(A_rows)
        world_coords = vh[-1]
        world_coords /= world_coords[-1]
        w[i] = world_coords.T

    # Back project to 2D homogeneous coordinates
    pts1_backprojected_homogeneous = (C1 @ w.T).T # N x 3
    pts2_backprojected_homogeneous = (C2 @ w.T).T # N x 3

    pts1_backprojected_homogeneous /= np.expand_dims(pts1_backprojected_homogeneous[:, -1], axis=-1)
    pts2_backprojected_homogeneous /= np.expand_dims(pts2_backprojected_homogeneous[:, -1], axis=-1)

    pts1_backprojected_homogeneous = pts1_backprojected_homogeneous[:, :-1] # N x 2
    pts2_backprojected_homogeneous = pts2_backprojected_homogeneous[:, :-1] # N x 2

    # Calculate reprojection error
    err = sum(np.linalg.norm(pts1 - pts1_backprojected_homogeneous, axis=1)**2) + sum(np.linalg.norm(pts2 - pts2_backprojected_homogeneous, axis=1)**2)
    P = w[:,:-1]

    return P, err


"""
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
"""


def findM2(F, pts1, pts2, intrinsics, filename="q3_3.npz"):
    """
    Q2.2: Function to find camera2's projective matrix given correspondences
        Input:  F, the pre-computed fundamental matrix
                pts1, the Nx2 matrix with the 2D image coordinates per row
                pts2, the Nx2 matrix with the 2D image coordinates per row
                intrinsics, the intrinsics of the cameras, load from the .npz file
                filename, the filename to store results
        Output: [M2, C2, P] the computed M2 (3x4) camera projective matrix, C2 (3x4) K2 * M2, and the 3D points P (Nx3)

    ***
    Hints:
    (1) Loop through the 'M2s' and use triangulate to calculate the 3D points and projection error. Keep track
        of the projection error through best_error and retain the best one.
    (2) Remember to take a look at camera2 to see how to correctly reterive the M2 matrix from 'M2s'.

    """
    # ----- TODO -----
    # YOUR CODE HERE
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    M1 = np.hstack((np.identity(3), np.zeros(3)[:, np.newaxis]))
    C1 = K1 @ M1
    
    E = essentialMatrix(F, K1, K2)
    M2s = camera2(E)

    true_M2 = None
    true_P = None
    max_num_pos = 0
    true_error = None
    true_C2 = None

    for i in range(M2s.shape[-1]):
        M2 = M2s[:, :, i]
        C2 = K2 @ M2
        P, reprojection_error = triangulate(C1, pts1, C2, pts2)
        num_pos = len(P[P[:,-1] > 0])
        if num_pos > max_num_pos:
            max_num_pos = num_pos
            true_M2 = M2
            true_P = P
            true_C2 = C2
            true_error = reprojection_error

    print(f"True Reprojection Error: {true_error}")

    return true_M2, true_C2, true_P


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    M2, C2, P = findM2(F, pts1, pts2, intrinsics)
    np.savez("q3_3.npz", M2, C2, P)
    # Simple Tests to verify your implementation:
    M1 = np.hstack((np.identity(3), np.zeros(3)[:, np.newaxis]))
    C1 = K1.dot(M1)
    C2 = K2.dot(M2)
    P_test, err = triangulate(C1, pts1, C2, pts2)
    assert err < 500
