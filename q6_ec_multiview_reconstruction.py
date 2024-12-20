import numpy as np
import matplotlib.pyplot as plt

import os

from helper import visualize_keypoints, plot_3d_keypoint, connections_3d, colors
from q3_2_triangulate import triangulate

# Insert your package here

"""
Q6.1 Multi-View Reconstruction of keypoints.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx3 matrix with the 2D image coordinates and confidence per row
            C2, the 3x4 camera matrix
            pts2, the Nx3 matrix with the 2D image coordinates and confidence per row
            C3, the 3x4 camera matrix
            pts3, the Nx3 matrix with the 2D image coordinates and confidence per row
    Output: P, the Nx3 matrix with the corresponding 3D points for each keypoint per row
            err, the reprojection error.

Modified by Vineet Tambe, 2023.
"""


def MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres=100):
    # TODO: Replace pass by your implementation

    inlier_indices = (pts1[:, -1] > Thres) & (pts2[:, -1] > Thres) & (pts3[:, -1] > Thres)
    pts1 = pts1[inlier_indices][:, :2]
    pts2 = pts2[inlier_indices][:, :2]
    pts3 = pts3[inlier_indices][:, :2]
    print(len(pts1), len(pts2), len(pts3))

    # Pts of the shape N x 2 (x, y)
    num_points = pts1.shape[0]

    # All ps are of the shape (1 x 4), equivalent to p11.T, pl2.T ...
    p11 = C1[0]; p12 = C1[1]; p13 = C1[2]
    p21 = C2[0]; p22 = C2[1]; p23 = C2[2]
    p31 = C3[0]; p32 = C2[1]; p33 = C3[2]

    w = np.zeros((num_points, 4))

    for i in range(num_points):
        # Extract relevant x and y points from pts1 and 2
        x1 = pts1[i][0]; x2 = pts2[i][0]; x3 = pts3[i][0]
        y1 = pts1[i][1]; y2 = pts2[i][1]; y3 = pts3[i][1]

        A_rows = np.array([y1*p13 - p12,
                           p11 - x1*p13,
                           y2*p23 - p22,
                           p21 - x2*p23,
                           y3*p33 - p32,
                           p31 - x3*p33])
        
        # Run SVD to get back w = [X, Y, Z, 1].T
        u, s, vh = np.linalg.svd(A_rows)
        world_coords = vh[-1]
        world_coords /= world_coords[-1]
        w[i] = world_coords.T

    # Back project to 2D homogeneous coordinates
    pts1_backprojected_homogeneous = (C1 @ w.T).T # N x 3
    pts2_backprojected_homogeneous = (C2 @ w.T).T # N x 3
    pts3_backprojected_homogeneous = (C3 @ w.T).T # N x 3

    pts1_backprojected_homogeneous /= np.expand_dims(pts1_backprojected_homogeneous[:, -1], axis=-1)
    pts2_backprojected_homogeneous /= np.expand_dims(pts2_backprojected_homogeneous[:, -1], axis=-1)
    pts3_backprojected_homogeneous /= np.expand_dims(pts3_backprojected_homogeneous[:, -1], axis=-1)

    pts1_backprojected_homogeneous = pts1_backprojected_homogeneous[:, :-1] # N x 2
    pts2_backprojected_homogeneous = pts2_backprojected_homogeneous[:, :-1] # N x 2
    pts3_backprojected_homogeneous = pts3_backprojected_homogeneous[:, :-1] # N x 2

    # Calculate reprojection error
    err = sum(np.linalg.norm(pts1 - pts1_backprojected_homogeneous, axis=1)**2) + sum(np.linalg.norm(pts2 - pts2_backprojected_homogeneous, axis=1)**2) + sum(np.linalg.norm(pts3-pts3_backprojected_homogeneous, axis=1)**2)
    P = w[:,:-1]

    return P, err


"""
Q6.2 Plot Spatio-temporal (3D) keypoints
    :param car_points: np.array points * 3
"""


def plot_3d_keypoint_video(pts_3d_video):
    # TODO: Replace pass by your implementation
    pass


# Extra Credit
if __name__ == "__main__":
    pts_3d_video = []
    for loop in range(10):
        print(f"processing time frame - {loop}")

        data_path = os.path.join("data/q6/", "time" + str(loop) + ".npz")
        image1_path = os.path.join("data/q6/", "cam1_time" + str(loop) + ".jpg")
        image2_path = os.path.join("data/q6/", "cam2_time" + str(loop) + ".jpg")
        image3_path = os.path.join("data/q6/", "cam3_time" + str(loop) + ".jpg")

        im1 = plt.imread(image1_path)
        im2 = plt.imread(image2_path)
        im3 = plt.imread(image3_path)

        data = np.load(data_path)
        pts1 = data["pts1"]
        pts2 = data["pts2"]
        pts3 = data["pts3"]

        K1 = data["K1"]
        K2 = data["K2"]
        K3 = data["K3"]

        M1 = data["M1"]
        M2 = data["M2"]
        M3 = data["M3"]

        # Note - Press 'Escape' key to exit img preview and loop further
        img = visualize_keypoints(im2, pts2)

        # TODO: YOUR CODE HERE
        C1 = K1 @ M1
        C2 = K2 @ M2
        C3 = K3 @ M3

        P, err = MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3)
        print(f"3D reconstruction error: {err}")
        plot_3d_keypoint(P)
        np.savez("q6_1.npz", P)
        