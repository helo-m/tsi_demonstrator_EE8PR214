import cv2 as cv
import glob
import numpy as np
import os

# Clear the console screen
os.system('cls' if os.name == 'nt' else 'clear')

def calibrate_camera(images):
    """
    Calibrates a single camera using a set of images.

    Parameters:
        images (list): List of image file paths.

    Returns:
        mtx (numpy.ndarray): Camera matrix.
        dist (numpy.ndarray): Distortion coefficients.
    """
    # Criteria used by the checkerboard pattern detector
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Checkerboard dimensions (inner corners)
    rows = 4
    columns = 5
    world_scaling = 1  # Real-world square size scaling factor

    # Coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # Lists to store object points and image points
    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane

    # Frame dimensions (assume all images are the same size)
    width = None
    height = None

    for frame_path in images:
        frame = cv.imread(frame_path)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

        if ret:
            # Refine corner detection
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)

            # Set frame dimensions
            if width is None or height is None:
                height, width = gray.shape

    # Perform camera calibration
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    return mtx, dist, ret


def stereo_calibrate(mtx1, dist1, mtx2, dist2, images_camera0, images_camera1):
    """
    Performs stereo calibration using two sets of images.

    Parameters:
        mtx1, dist1: Camera matrix and distortion coefficients for the first camera.
        mtx2, dist2: Camera matrix and distortion coefficients for the second camera.
        images_camera0 (list): List of image file paths for the first camera.
        images_camera1 (list): List of image file paths for the second camera.

    Returns:
        R (numpy.ndarray): Rotation matrix.
        T (numpy.ndarray): Translation vector.
        ret (float): Stereo calibration RMSE.
    """
    # Criteria for stereo calibration
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    # Checkerboard dimensions (inner corners)
    rows = 4
    columns = 5
    world_scaling = 1.0

    # Coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # Lists to store object points and image points
    objpoints = []  # 3D points in real-world space
    imgpoints_left = []  # 2D points in image plane for the first camera
    imgpoints_right = []  # 2D points in image plane for the second camera

    # Frame dimensions (assume all images are the same size)
    width = None
    height = None

    for frame1_path, frame2_path in zip(images_camera0, images_camera1):
        frame1 = cv.imread(frame1_path)
        frame2 = cv.imread(frame2_path)

        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

        # Find the checkerboard in both images
        ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)

        if ret1 and ret2:
            # Refine corner detection
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

            # Set frame dimensions
            if width is None or height is None:
                height, width = gray1.shape

    # Perform stereo calibration
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right, mtx1, dist1, mtx2, dist2,
        (width, height), criteria=criteria, flags=stereocalibration_flags
    )

    return R, T, ret
