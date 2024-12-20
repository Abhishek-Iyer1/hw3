import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sympy import Symbol, solve_poly_system, det, Matrix, MatrixSymbol
from helper import displayEpipolarF, calc_epi_error, toHomogenous, _singularize, refineF

# Insert your package here


"""
Q2.2: Seven Point Algorithm for calculating the fundamental matrix
    Input:  pts1, 7x2 Matrix containing the corresponding points from image1
            pts2, 7x2 Matrix containing the corresponding points from image2
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated 3x3 fundamental matrixes.
    
    HINTS:
    (1) Normalize the input pts1 and pts2 scale paramter M.
    (2) Setup the seven point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Pick the last two colum vector of vT.T (the two null space solution f1 and f2)
    (5) Use the singularity constraint to solve for the cubic polynomial equation of  F = a*f1 + (1-a)*f2 that leads to 
        det(F) = 0. Solving this polynomial will give you one or three real solutions of the fundamental matrix. 
        Use np.polynomial.polynomial.polyroots to solve for the roots
    (6) Unscale the fundamental matrixes and return as Farray
"""


def sevenpoint(pts1, pts2, M):
    Farray = []
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

    f1 = vh[-1]
    f2 = vh[-2]

    f1 = f1.reshape(3, 3)
    f2 = f2.reshape(3, 3)

    a = Symbol('a')
    polynomial_a = Matrix((a*f1) + ((1-a)*f2)).det()
    coeffs = polynomial_a.as_poly(a).all_coeffs()
    coeffs_float = [float(coeff) for coeff in coeffs]
    solutions = np.polynomial.polynomial.polyroots(coeffs_float)
    Farray = np.zeros((len(solutions),3,3))
    
    for i, a in enumerate(solutions):
        F_el = ((a*f1) + ((1-a)*f2)).T
        F_el_enforced = _singularize(F_el)
        F_el_unscaled = T.T @ F_el_enforced @ T
        F_el_unscaled /= F_el_unscaled[2, 2]
        Farray[i] = F_el_unscaled
        
    return Farray


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    # indices = np.arange(pts1.shape[0])
    # indices = np.random.choice(indices, 7, False)
    indices = np.array([82, 19, 56, 84, 54, 24, 18])

    M = np.max([*im1.shape, *im2.shape])

    Farray = sevenpoint(pts1[indices, :], pts2[indices, :], M)

    print(Farray)

    # fundamental matrix must have rank 2!
    # assert(np.linalg.matrix_rank(F) == 2)

    # Simple Tests to verify your implementation:
    # Test out the seven-point algorithm by randomly sampling 7 points and finding the best solution.
    np.random.seed(1)  # Added for testing, can be commented out

    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    max_iter = 500
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    ress = []
    F_res = []
    choices = []
    M = np.max([*im1.shape, *im2.shape])
    for i in tqdm(range(max_iter)):
        choice = np.random.choice(range(pts1.shape[0]), 7)
        pts1_choice = pts1[choice, :]
        pts2_choice = pts2[choice, :]
        Fs = sevenpoint(pts1_choice, pts2_choice, M)
        for F in Fs:
            choices.append(choice)
            res = calc_epi_error(pts1_homo, pts2_homo, F)
            F_res.append(F)
            ress.append(np.mean(res))

    min_idx = np.argmin(np.abs(np.array(ress)))
    F = F_res[min_idx]
    np.savez("q2_2.npz", F, M)
    print(F, M)
    displayEpipolarF(im1, im2, F)
    print("Error:", ress[min_idx])

    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2
    assert np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1
