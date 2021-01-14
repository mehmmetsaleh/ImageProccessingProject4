import unittest

import sol4 as sol
from imageio import imread
from skimage.color import rgb2gray
import numpy as np
import os
import inspect
import runner as run
import ast

import cv2
from scipy.stats import pearsonr
import sol4_utils
import matplotlib.pyplot as plt


# ================================ helper functions ================================


def read_image(filename, representation):
    """
    Receives an image file and converts it into one of two given representations.
    :param filename: The file name of an image on disk (could be grayscale or RGB).
    :param representation: representation code, either 1 or 2 defining wether the output
    should be a grayscale image (1) or an RGB image (2). If the input image is grayscale,
    we won't call it with representation = 2.
    :return: An image, represented by a matrix of type (np.float64) with intensities
    normalized to the range [0,1].
    """
    assert representation in [1, 2]

    # reads the image
    im = imread(filename)
    if representation == 1:  # If the user specified they need grayscale image,
        if len(im.shape) == 3:  # AND the image is not grayscale yet
            im = rgb2gray(im)  # convert to grayscale (**Assuming its RGB and not a different format**)

    im_float = im.astype(np.float64)  # Convert the image type to one we can work with.

    if im_float.max() > 1:  # If image values are out of bound, normalize them.
        im_float = im_float / 255

    return im_float


def _mse(im1, im2):
    """
    Calculates 'Mean Squared Error' between the two similar shaped images, which is the sum of the squared difference
    between the two images. Normalizes the MSE.
    The lower the error the more similar the images are.
    :param im1: First image.
    :param im2: Second image.
    :return: err: The error.
    """
    err = np.sum((im1.astype("float") - im2.astype("float")) ** 2)
    err /= float(im1.shape[0] * im1.shape[1])

    return err


def _generate_images(directory_path):
    """
    Generates a list of images from a list of image names.
    :param names: List of strings.
    :return: A list of grayscale images.
    """
    directory = os.path.abspath(directory_path)
    images = [(read_image(os.path.join(directory, filename), 1), filename) for filename in os.listdir(directory) if
              filename.endswith('.jpg')]
    return images


def _does_contain(function, statements):
    """
    Checks if a function implementation contains any usage of given tokens.
    :param function: The function to check in.
    :param statements: The statement tokens to find.
    :return: True if there is an instance of the statements in the function implementation, False otherwise.
    """
    nodes = ast.walk(ast.parse(inspect.getsource(function)))
    return any(isinstance(node, statements) for node in nodes)


def _uses_loop(function):
    """
    Checks if a function uses top level loops.
    :param function: The function to check in.
    :return: True if it contains loops, False otherwise.
    """
    loop_statements = ast.For, ast.While, ast.AsyncFor
    return _does_contain(function, loop_statements)


def _has_return(function):
    """
    Checks if a function contains a return statement.
    :param function: The function to check in.
    :return: True if it contains a return statement, False otherwise.
    """
    return _does_contain(function, ast.Return)


# -------------------------------- ex3-specifics --------------------------------


def _cv2_build_gaussian_pyramid(im, levels):
    """
    Builds a gaussian pyramid for a given image using the built in reduce function in cv2.
    :param im: The given image.
    :param levels: Amount of levels for the pyramid.
    :return: gaussian_pyr: The gaussian pyramid.
    """
    layer = im.copy()
    # Building a gaussian pyramid using a cv2 builtIn capabilities
    gaussian_pyr = []
    gaussian_pyr.append(np.array(im))
    for i in range(levels - 1):
        layer = np.array(cv2.pyrDown(layer))
        gaussian_pyr.append(layer)
    return gaussian_pyr


def _cv2_build_laplacian_pyramid(gaussian_pyr):
    """
    Builds a laplacian pyramid from a given gaussian pyramid using cv2 built in expand.
    :param gaussian_pyr: The given gaussian pyramid.
    :return: The laplacian pyramid.
    """
    # Computing a laplacian pyramid using the gaussian pyramid created using cv2
    laplacian_pyr = []
    laplacian_pyr.append(gaussian_pyr[-1])
    for i in range(len(gaussian_pyr) - 1, 0, -1):
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
        laplacian_layer = cv2.subtract(gaussian_pyr[i - 1], gaussian_expanded)
        laplacian_pyr.append(laplacian_layer)
    laplacian_pyr = laplacian_pyr[::-1]
    return laplacian_pyr


# ================================ unittest class ================================


class TestEx4(unittest.TestCase):
    """
    The unittest testing suite.
    """

    # ================================ setup/teardown functions ================================

    @classmethod
    def setUpClass(cls):
        """
        Generates all necessary data for tests, runs before all other tests.
        :return: -
        """
        # cls.images = _generate_images(r'external')
        # cls.filter_sizes = [3, 5, 7, 9]

    # ================================ general helpers ================================

    def _structure_tester(self, func, signature, no_loops, no_return):
        """
        Checks a given function's structure is correct according to the pdf.
        :param func: The given function.
        :param signature: Expected signature.
        :param no_loops: True if there should be no loops.
        :param no_return: True if the function returns nothing.
        :return: -
        """
        func_name = str(func.__name__)

        # Check no loops were used in implementation if needed
        if no_loops:
            self.assertEqual(False, _uses_loop(func),
                             msg=f"Your {func_name} implementation should not contain loops")

        # Check there is no return statement if needed
        if no_return:
            self.assertEqual(False, _has_return(func),
                             msg=f"Your {func_name} implementation should not have a return statement")

        # Checks the signature of the function equals the pdf
        self.assertEqual(signature, str(inspect.signature(func)),
                         msg=f"{func_name} signature should be {signature} but is {str(inspect.signature(func))}")

    def _compare_images(self, expected_im, sol_image, tested_im_name, tested_func_name, pearson_thresh=0.9,
                        mse_thresh=0.05):
        """
        Compares two images by first comparing their shape, and then checking their similarities by checking the pearson's
        "r" coefficient is higher than "pearson_tresh" and the mse error is lower than "mse_tresh".
        :param expected_im: The reference image.
        :param sol_image: The tested image.
        :param tested_im_name: Name of the image being tested.
        :param tested_func_name: Name of the function that changed the image.
        :param pearson_thresh: The pearson's r coefficient threshold.
        :param mse_thresh: The mse error threshold.
        :return:
        """
        self.assertEqual(expected_im.shape, sol_image.shape,
                         msg=f"The '{tested_func_name}' function on the {tested_im_name} image should be similar to the built in output, so the output's shape should be equal to the shape of the built in shape")
        r = pearsonr(expected_im.flatten(), sol_image.flatten())[0]

        self.assertTrue(r > pearson_thresh and _mse(expected_im, sol_image) < mse_thresh,
                        msg=f"The {tested_im_name} image from {tested_func_name}'s output is not so similar to the built in implementation... maybe you should used plt.imshow on the new image and see what it looks like")

    # ================================ Part 3.1 Tests ================================

    # -------------------------------- helpers --------------------------------

    def _get_random_coords(self, start, end, pad_start, pad_end):
        x_start, y_start = np.random.randint(start, end, 2)
        x_end = np.random.randint(x_start + pad_start, end + pad_end)
        y_end = np.random.randint(y_start + pad_start, end + pad_end)
        return x_start, x_end, y_start, y_end

    def _fill(self, A, B, fill_below=True, xs=10, ys=12):
        m = (B[1] - A[1]) / (B[0] - A[0])
        b = A[1] - m * A[0]

        Y, X = np.mgrid[0:ys, 0:xs]
        L = m * X + b - Y

        op = np.greater_equal if fill_below else np.less_equal

        return op(L, 0.0)

    def _get_rhombus_corners(self):
        start = np.random.randint(10, 500)
        diff = np.random.randint(5, 200)
        second = start + diff
        third = start + (diff * 2)
        corners = ((start, second), (second, third), (third, second), (second, start))
        return corners

    # -------------------------------- harris test --------------------------------

    def _harris_rhombus_module(self, corners, negative=False):
        # draw rhombus
        rhombus = self._fill(corners[0], corners[1], True, 1024, 1024) & \
                  self._fill(corners[1], corners[2], True, 1024, 1024) & \
                  self._fill(corners[2], corners[3], False, 1024, 1024) & \
                  self._fill(corners[3], corners[0], False, 1024, 1024)
        if not negative:
            rhombus = 1 - rhombus

        # get output
        res = sol.harris_corner_detector(rhombus)

        self.assertEqual((4, 2), np.array(res).shape,
                         msg=f"harris should return an array of shape (4, 2) on a single rhombus")

        # Checks the corners or something close to it (by a margin of 1 for each axis) is in the result
        res = {tuple(p) for p in res}
        expected = set(corners)
        for c in expected:
            self.assertTrue(
                c in res or (c[0] - 1, c[1]) in res or (c[0], c[1] - 1) in res or (c[0] - 1, c[1] - 1) in res or (
                    c[0] + 1, c[1]) in res or (c[0], c[1] + 1) in res or (c[0] + 1, c[1] + 1) in res,
                msg=f"The expected corner {c} or any similar corner was not found by harris on rhombus")

    def _harris_square_module(self, bounds, bounds2=None, negative=False):

        # Draws a rectangle/2 rectangles according to the given bounds
        if not negative:
            x_start, x_end, y_start, y_end = bounds
            mat = np.ones((1024, 1024)) * 255
            mat[x_start: x_end, y_start: y_end] = np.zeros((x_end - x_start, y_end - y_start))
            if bounds2 is not None:
                x_start2, x_end2, y_start2, y_end2 = bounds2
                mat[x_start2: x_end2, y_start2: y_end2] = np.zeros((x_end2 - x_start2, y_end2 - y_start2))
        # Creates the rectangle in negative colors
        else:
            mat = np.zeros((1024, 1024))
            x_start, x_end, y_start, y_end = bounds
            mat[x_start: x_end, y_start: y_end] = np.ones((x_end - x_start, y_end - y_start)) * 255

        # get output
        res = sol.harris_corner_detector(mat)

        # Checks output's shape
        if bounds2 is None:
            self.assertEqual((4, 2), np.array(res).shape,
                             msg=f"harris should return an array of shape (4, 2) on a single rectangle")
        else:
            self.assertEqual((8, 2), np.array(res).shape,
                             msg=f"harris should return an array of shape (8, 2) on two rectangles")
        # Checks output's type
        self.assertTrue(np.issubdtype(np.array(res).dtype, np.integer), msg=f"harris's output should be arrays of ints")

        # Check all corners were found
        res = {tuple(p) for p in res}
        expected = {(y_start, x_start), (y_end - 1, x_start), (y_start, x_end - 1), (y_end - 1, x_end - 1)}
        if bounds2 is not None:
            expected.union(
                {(y_start2, x_start2), (y_end2 - 1, x_start2), (y_start2, x_end2 - 1), (y_end2 - 1, x_end2 - 1)})

        self.assertTrue(expected.issubset(res),
                        msg=f"Failed on the square test on bounds : {bounds}. The corners : {set(expected).difference(res)} were not in the found corners")

    def test_harris_basic(self):

        # Test basic structure
        self._structure_tester(sol.harris_corner_detector, r'(im)', False, False)

        # test for 1 random rectangle
        for i in range(20):
            self._harris_square_module(self._get_random_coords(1, 1000, 7, 20))

        # test for 1 random negative rectangle
        for i in range(20):
            self._harris_square_module(self._get_random_coords(1, 1000, 7, 20), negative=True)

        # test for 2 random rectangles
        for i in range(20):
            self._harris_square_module(self._get_random_coords(1, 492, 7, 20),
                                       self._get_random_coords(513, 1000, 7, 20))

        # test for 1 semi-random rhombus
        for i in range(20):
            self._harris_rhombus_module(self._get_rhombus_corners())

        # test for 1 semi-random negative rhombus
        for i in range(20):
            self._harris_rhombus_module(self._get_rhombus_corners(), negative=True)

    # -------------------------------- sample_decriptor test --------------------------------

    def _test_sample_module(self, im, pos, rad, expected):

        # Get result
        res = np.array(sol.sample_descriptor(np.array(im), np.array(pos), rad))

        # Checks output's shape and dtype
        K = rad * 2 + 1
        N = len(pos)
        self.assertEqual(f"({N}, {K}, {K})", str(res.shape),
                         msg=f"Shape of 'sample_decriptor' output should be (N,K,K)")
        self.assertEqual(np.dtype('float64'), res.dtype, msg=f"The presubmit expects the output to be float64")

        # Compares output to my output
        self.assertIsNone(np.testing.assert_array_almost_equal(expected, res, decimal=5,
                                                               err_msg=f"Descriptor is different from mine, maybe there is a problem?"))

    def test_sample_decriptor_basic_COMPARE_TO_MY_OUTPUT(self):

        # Check structure
        self._structure_tester(sol.sample_descriptor, r'(im, pos, desc_rad)', False, False)

        # Compares sample_descriptor output to my output on 10 rectangles
        for i in range(1, 11):
            mat = np.load(os.path.abspath(f'test_data/testing_arrays/rectangles/mat{i}.npy'))
            pos = np.load(os.path.abspath(f'test_data/testing_arrays/rectangles/pos{i}.npy'))
            expected = np.load(os.path.abspath(f'test_data/testing_arrays/rectangles/res{i}.npy'))
            self._test_sample_module(mat, pos, 3, expected)

        # Compares sample_descriptor output to my output on 10 diamonds
        for i in range(1, 6):
            mat = np.load(os.path.abspath(f'test_data/testing_arrays/rhombus/mat{i}.npy'))
            pos = np.load(os.path.abspath(f'test_data/testing_arrays/rhombus/pos{i}.npy'))
            expected = np.load(os.path.abspath(f'test_data/testing_arrays/rhombus/res{i}.npy'))
            self._test_sample_module(mat, pos, 3, expected)

        # Compares sample_descriptor output to my output on 3 images: Basic chess board and the two oxford images
        names = ["chess", "oxford1", "oxford2"]
        for name in names:
            mat = np.load(os.path.abspath(f'test_data/testing_arrays/images/{name}.npy'))
            pos = np.load(os.path.abspath(f'test_data/testing_arrays/images/{name}_pos1.npy'))
            expected = np.load(os.path.abspath(f'test_data/testing_arrays/images/{name}_res1.npy'))
            self._test_sample_module(mat, pos, 3, expected)

    # -------------------------------- structure test for 'find_features' --------------------------------

    def test_structure_find_features(self):

        # Check structure
        self._structure_tester(sol.find_features, r'(pyr)', False, False)

        # Checks 'spread_out_corners' was used instead of 'harris_corner_detector'
        find_features_text = inspect.getsource(sol.find_features)
        self.assertTrue(r"spread_out_corners(" in find_features_text,
                        msg=f"You should use 'spread_out_corners' and not use 'harris_corner_detector' in 'find_features' function")
        self.assertTrue(r"harris_corner_detector(" not in find_features_text,
                        msg=f"You should use 'spread_out_corners' and not use 'harris_corner_detector' in 'find_features' function")

        # check return shape
        res = sol.find_features([np.ones((1024, 1024)), np.ones((512, 512)), np.ones((256, 256)), np.ones((128, 128))])
        self.assertEqual(2, len(res), msg=f"'find_features' function's output should be an array of length 2")


if __name__ == '__main__':
    runner = run.CustomTextTestRunner()
    unittest.main(testRunner=runner)
