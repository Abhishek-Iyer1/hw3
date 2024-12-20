import numpy as np
import matplotlib.pyplot as plt

from helper import _epipoles

from q2_1_eightpoint import eightpoint

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
def epipolarMatchGUI(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title("Select a point in this image")
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title(
        "Verify that the corresponding point \n is on the epipolar line in this image"
    )
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        # x, y = plt.ginput(1, mouse_stop=2)[0]

        out = plt.ginput(1, timeout=3600, mouse_stop=2)

        if len(out) == 0:
            print(f"Closing GUI")
            break

        x, y = out[0]

        xc = int(x)
        yc = int(y)
        v = np.array([xc, yc, 1])
        l = F.dot(v)
        s = np.sqrt(l[0] ** 2 + l[1] ** 2)

        if s == 0:
            print("Zero line vector in displayEpipolar")

        l = l / s

        if l[0] != 0:
            ye = sy - 1
            ys = 0
            xe = -(l[1] * ye + l[2]) / l[0]
            xs = -(l[1] * ys + l[2]) / l[0]
        else:
            xe = sx - 1
            xs = 0
            ye = -(l[0] * xe + l[2]) / l[1]
            ys = -(l[0] * xs + l[2]) / l[1]

        # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2);
        ax1.plot(x, y, "*", markersize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)

        # draw points
        x2, y2 = epipolarCorrespondence(I1, I2, F, xc, yc)
        ax2.plot(x2, y2, "ro", markersize=8, linewidth=2)
        plt.draw()


"""
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2
            
    Hints:
    (1) Given input [x1, x2], use the fundamental matrix to recover the corresponding epipolar line on image2
    (2) Search along this line to check nearby pixel intensity (you can define a search window) to 
        find the best matches
    (3) Use guassian weighting to weight the pixel simlairty

"""

def kernel(x1, y1, kernel_size):
    x1_kernel_list = np.arange(x1, x1+kernel_size)
    y1_kernel_list = np.arange(y1, y1+kernel_size)
    x, y = np.meshgrid(x1_kernel_list, y1_kernel_list)
    x = x.flatten(); y = y.flatten()
    x_centered = x - kernel_size // 2
    y_centered = y - kernel_size // 2
    return y_centered, x_centered

def generate_gaussian(kernel_size):
    gaussian = np.array([[1, 2, 4, 2, 1],
                [2, 4, 8, 4, 2],
                [4, 8, 16, 8, 4],
                [2, 4, 8, 4, 2],
                [1, 2, 4, 2, 1]])
    
    return np.expand_dims(gaussian/100, axis=-1)

def pad_image(img, kernel_size):
    w, h, c = img.shape
    padded_img = np.ones((w+kernel_size-1, h+kernel_size-1, c)) * 255
    padded_img[kernel_size//2:-(kernel_size//2), kernel_size//2:-(kernel_size//2)] = img
    return padded_img

def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    # ----- TODO -----
    # YOUR CODE HERE

    kernel_size = 5
    im1_padded = pad_image(im1, kernel_size)
    im2_padded = pad_image(im2, kernel_size)

    # Construct right epipolar line using F
    f11, f12, f13, f21, f22, f23, f31, f32, f33 = F.flatten()
    al = (f11*x1 + f21*y1 + f31)
    bl = (f12*x1 + f22*y1 + f32)
    cl = (f13*x1 + f23*y1 + f33)

    slope = -1*al / bl
    intercept = -1*cl / bl

    local_search = 40
    y2_array = np.arange(y1-local_search, y1+local_search)
    x2_array = (y2_array - intercept)/slope
    x2_array = x2_array.astype(int)

    # Define Kernel size and weightings
    x1_shifted = x1 + kernel_size//2
    y1_shifted = y1 + kernel_size//2
    im1_kernel = kernel(x1_shifted, y1_shifted, kernel_size)
    gaussian = generate_gaussian(kernel_size)
    im1_neighbour = im1_padded[im1_kernel].reshape((kernel_size, kernel_size, 3))
    im1_window = gaussian * im1_neighbour

    lowest_error = np.inf
    matching_point = None

    for x2,y2 in list(zip(x2_array, y2_array)):
        x2_shifted = x2 + kernel_size//2
        y2_shifted = y2 + kernel_size//2
        im2_kernel = kernel(x2_shifted, y2_shifted, kernel_size)
        im2_neighbour = im2_padded[im2_kernel].reshape((kernel_size, kernel_size, 3))
        im2_window = gaussian * im2_neighbour
        curr_err = np.sum(np.linalg.norm(im2_window - im1_window, axis=0))
        if curr_err < lowest_error:
            lowest_error = curr_err
            matching_point = (x2, y2)

    return matching_point

if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the cameras
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    np.savez("q4_1.npz", F, pts1, pts2)
    epipolarMatchGUI(im1, im2, F)

    # Simple Tests to verify your implementation:
    x2, y2 = epipolarCorrespondence(im1, im2, F, 119, 217)
    print(x2, y2)
    assert np.linalg.norm(np.array([x2, y2]) - np.array([118, 181])) < 10
