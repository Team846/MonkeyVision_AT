from typing import Tuple
import cv2
import numpy as np

def GET_CAMERA_ANGLES(
    points: np.ndarray,
    frame: cv2.typing.MatLike,
    dist_coeffs: np.ndarray,
    cam_mtx: np.ndarray,
    cam_fov_x: float,
    cam_fov_y: float,
) -> np.ndarray:
    
    img_points = points.astype(np.float32).reshape(-1, 1, 2)
    normalized_points = cv2.undistortPoints(img_points, cam_mtx, dist_coeffs, None, cam_mtx)
    
    x_normalized = normalized_points[:, 0, 0]
    y_normalized = normalized_points[:, 0, 1]
    
    angles = np.empty((len(points), 2), dtype=np.float64)
    
    # TODO: check if the *2 is real
    angle_x = (x_normalized - frame.shape[1] / 2.0) / frame.shape[1] * cam_fov_x*2
    angle_y = (-y_normalized + frame.shape[0] / 2.0) / frame.shape[0] * cam_fov_y
    
    angles[:, 0] = angle_x
    angles[:, 1] = angle_y
    
    return angles
