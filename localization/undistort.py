from typing import Tuple
import cv2
import numpy as np

CAM_APT_X = 8.7586
CAM_APT_Y = 8.7586


def GET_CAMERA_ANGLES(
    x_img: float,
    y_img: float,
    frame: cv2.typing.MatLike,
    dist_coeffs: np.ndarray,
    cam_mtx: np.ndarray,
) -> Tuple[float, float]:

    img_points = np.array([[x_img, y_img]], dtype="float32")
    normalized_points = cv2.undistortPoints(
        img_points, cam_mtx, dist_coeffs, None, cam_mtx
    )

    x_normalized = normalized_points[0][0][0]
    y_normalized = normalized_points[0][0][1]

    cam_fov_x, cam_fov_y, _, _, _ = cv2.calibrationMatrixValues(
        cam_mtx, frame.shape, CAM_APT_X, CAM_APT_Y
    )

    angle_x = (x_normalized - frame.shape[1] / 2.0) / frame.shape[1] * cam_fov_x * 2.0
    angle_y = (y_normalized - frame.shape[0] / 2.0) / frame.shape[0] * cam_fov_y

    return angle_x, angle_y
