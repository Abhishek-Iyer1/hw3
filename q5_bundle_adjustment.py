import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

from helper import displayEpipolarF, calc_epi_error, toHomogenous
from q2_1_eightpoint import eightpoint
from q2_2_sevenpoint import sevenpoint
from q3_2_triangulate import findM2
from tqdm import tqdm
import math
import scipy

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""


def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:, 0], P_before[:, 1], P_before[:, 2], c="blue")
    ax.scatter(P_after[:, 0], P_after[:, 1], P_after[:, 2], c="red")
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()


"""
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on 
        the results/expected number of inliners. You can also define your own metric. 
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values
 
"""


def ransacF(pts1, pts2, M, nIters=1000, tol=10):

    best_F = None
    inliers = None
    max_inliers = 0

    pts1_homogenous = np.hstack((pts1, np.ones((pts1.shape[0], 1)))) # N x 3
    pts2_homogenous = np.hstack((pts2, np.ones((pts2.shape[0], 1)))) # N x 3

    all_indices = np.arange(0, len(pts1_homogenous))

    for _ in tqdm(range(nIters)):

        indices = np.random.choice(all_indices, 8, replace = False)

        F = eightpoint(pts1[indices], pts2[indices], M)

        indices_inliers = all_indices[calc_epi_error(pts1_homogenous, pts2_homogenous, F) < tol]

        if len(indices_inliers) > max_inliers:
            best_F = F
            inliers = indices_inliers
            max_inliers = len(indices_inliers)
    
    best_F = eightpoint(pts1[inliers], pts2[inliers], M)
    best_indices_inliers = [calc_epi_error(pts1_homogenous, pts2_homogenous, best_F) < tol]

    return best_F, best_indices_inliers[0]


"""
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
"""


def rodrigues(r):
    # TODO: Replace pass by your implementation
    theta = np.linalg.norm(r)

    if theta == 0:
        R = np.eye(3)
    else:
        u = r/theta
        ux = np.array([[0, -u[2], u[1]],
                       [u[2], 0, -u[0]],
                       [-u[1], u[0], 0]])
        R = np.eye(3) + np.sin(theta)*ux + (1-np.cos(theta))*(ux@ux)

    return R


"""
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
"""


def invRodrigues(R):
    # TODO: Replace pass by your implementation
    A = (R - R.T)/2
    rho = np.array([A[2,1], A[0,2], A[1,0]]).T
    s = np.linalg.norm(rho)
    c = (R[0,0] + R[1,1] + R[2,2] - 1)/2

    if s == 0 and c == 1:
        r = np.zeros((3, 1))

    elif s == 0 and c == -1:
        for i in range(3):
            if not np.all(R[:, i] == np.array([0, 0, 0])):
                v = R[:, i]
                break

        u = v/np.linalg.norm(v)
        r = u*np.pi

        if (math.isclose((np.linalg.norm(r) == np.pi), math.pi) and ((r[0] == 0 and r[1] == 0) and r[2] < 0) or (r[0] == 0 and r[1] < 0) or (r[0] < 0)):
            r = -r
        else:
            r = r

    else:
        u = rho / s
        theta = np.arctan2(s, c)
        r = u*theta

    return r



"""
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
"""


def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # TODO: Replace pass by your implementation
    w = x[:-6]
    r2 = x[-6:-3]
    t2 = x[-3:]

    M2 = np.hstack((rodrigues(r2), np.expand_dims(t2, axis=-1)))
    
    C1 = K1 @ M1
    C2 = K2 @ M2

    w = np.reshape(w, (-1, 3)) # N x 3

    w_homogenous = np.hstack((w, np.ones((w.shape[0], 1))))
    pts1_homogenous_predicted = (C1 @ w_homogenous.T).T
    pts2_homogenous_predicted = (C2 @ w_homogenous.T).T

    pts1_hat = (pts1_homogenous_predicted / np.expand_dims(pts1_homogenous_predicted[:, -1], axis=-1))[:, :2]
    pts2_hat = (pts2_homogenous_predicted / np.expand_dims(pts2_homogenous_predicted[:, -1], axis=-1))[:, :2]

    residuals = np.concatenate([(p1 - pts1_hat).reshape([-1]), (p2 - pts2_hat).reshape([-1])]) # 4N x 1

    return residuals


"""
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual. 
        You can try different (method='..') in scipy.optimize.minimize for best results. 
"""

def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    obj_start = obj_end = 0
    # ----- TODO -----
    # YOUR CODE HERE
    R2 = M2_init[:, :3]
    t2 = M2_init[:, -1]

    r_vec = invRodrigues(R2)
    flattened_world_coords = P_init.flatten()

    x0 = np.array([*flattened_world_coords, *r_vec, *t2])

    def optimization_function(x):
        residuals = rodriguesResidual(K1, M1, p1, K2, p2, x)
        return np.linalg.norm(residuals)**2
    
    x_optimal = scipy.optimize.minimize(fun=optimization_function, x0=x0)

    w = x_optimal.x[:-6]
    r2 = x_optimal.x[-6:-3]
    t2 = x_optimal.x[-3:]

    M2 = np.hstack((rodrigues(r2), np.expand_dims(t2, axis=-1)))
    P = np.reshape(w, (-1, 3))

    obj_start = optimization_function(x0)
    obj_end = optimization_function(x_optimal.x)

    return M2, P, obj_start, obj_end


if __name__ == "__main__":
    np.random.seed(1)  # Added for testing, can be commented out

    some_corresp_noisy = np.load(
        "data/some_corresp_noisy.npz"
    )  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    noisy_pts1, noisy_pts2 = some_corresp_noisy["pts1"], some_corresp_noisy["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    #########################################
    F_init = eightpoint(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]))
    F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]))
    print(F, F_init)
    print(np.sum(inliers) / len(inliers))
    displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(noisy_pts1), toHomogenous(
        noisy_pts2
    )
    print(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)))
    print(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F_init)))
    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2
    quit()
    ##########################################

    # Simple Tests to verify your implementation:
    from scipy.spatial.transform import Rotation as sRot

    rotVec = sRot.random()
    mat = rodrigues(rotVec.as_rotvec())
    print(f"Error for rodrigues: {np.linalg.norm(rotVec.as_matrix() - mat)}")
    print(f"Error for inverse rodrigues: {np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat))}")
    assert np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)) < 1e-3
    assert np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3

    # Visualization:
    np.random.seed(1)
    correspondence = np.load(
        "data/some_corresp_noisy.npz"
    )  # Loading noisy correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")
    M = np.max([*im1.shape, *im2.shape])

    # TODO: YOUR CODE HERE
    """
    Call the ransacF function to find the fundamental matrix
    Call the findM2 function to find the extrinsics of the second camera
    Call the bundleAdjustment function to optimize the extrinsics and 3D points
    Plot the 3D points before and after bundle adjustment using the plot_3D_dual function
    """
    M1 = np.hstack((np.identity(3), np.zeros(3)[:, np.newaxis]))
    F, inliers_indices = ransacF(pts1, pts2, M)
    print(len(inliers_indices))
    M2, C2, P_before = findM2(F,pts1[inliers_indices], pts2[inliers_indices], intrinsics)
    M2, P_after, obj_start, obj_end = bundleAdjustment(K1, M1, pts1[inliers_indices], K2, M2, pts2[inliers_indices], P_before)
    print(obj_start, obj_end)
    plot_3D_dual(P_before, P_after)