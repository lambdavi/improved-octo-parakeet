from graspnet.taskgraspnet import TaskGraspNet
import torch
import numpy as np



def extract_intrinsics(camera_matrices):
    """
    Extracts fx, fy, cx, cy from CameraMatrices.

    Args:
        camera_matrices (CameraMatrices): An instance of CameraMatrices.

    Returns:
        fx, fy, cx, cy
    """
    # Extract from image matrix
    c_x = camera_matrices.image[0, 2]  # Principal point x
    c_y = camera_matrices.image[1, 2]  # Principal point y

    # Extract from focal matrix
    f_x = camera_matrices.focal[0, 0]  # Focal length in x
    f_y = camera_matrices.focal[1, 1]  # Focal length in y

    return f_x, f_y, c_x, c_y

