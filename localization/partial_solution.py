import cv2
import numpy as np
from localization.undistort import GET_CAMERA_ANGLES
from util.config import ConfigCategory, Config
from util.logger import Logger
from cv2.typing import MatLike
from typing import List
import math

import json
import os

TAG_FILE_PATH = "tags.json"

logger = Logger("PartialSolution")

loaded_tags = {}
if not os.path.exists(TAG_FILE_PATH):
    logger.Warn("Couldn't find existing AprilTag data file")
with open(TAG_FILE_PATH, "r") as tag_file:
    loaded_tags = json.load(tag_file)
    logger.Log(f"Loaded tags: {loaded_tags}")


CAM_FOV_X = 0
CAM_FOV_Y = 0
CAM_ANGLE_H = 0
CAM_ANGLE_V = 0
CAM_H = 0
CAM_X = 0
CAM_Y = 0


def SET_CAM(pipeline: int):

    pref_category = ConfigCategory(f"PartialSolution{pipeline}")
    global CAM_FOV_X, CAM_FOV_Y, CAM_ANGLE_H, CAM_ANGLE_V, CAM_H, CAM_X, CAM_Y
    CAM_FOV_X = pref_category.getFloatConfig("CAM_FOV_X_deg", 70.0)
    CAM_FOV_Y = pref_category.getFloatConfig("CAM_FOV_Y_deg", 47.27)
    CAM_ANGLE_H = pref_category.getFloatConfig("CAM_MOUNT_H_deg", 0.0)
    CAM_ANGLE_V = pref_category.getFloatConfig("CAM_MOUNT_V_deg", 0.0)
    CAM_H = pref_category.getFloatConfig("CAM_H_in", 12.0)
    CAM_X = pref_category.getFloatConfig("CAM_X_in", 0.0)
    CAM_Y = pref_category.getFloatConfig("CAM_Y_in", 0.0)


class Detection:
    def __init__(self, r_ground: float, theta_h: float, tag: int):
        self.r = r_ground
        self.theta = theta_h
        self.tag = tag

    def getTag(self) -> int:
        return self.tag

    def getR(self) -> float:
        return self.r

    def getTheta(self) -> float:
        return self.theta

    def __str__(self) -> str:
        return f"Detection Tag {self.tag}: ({self.r:.2f}, {self.theta:.2f} deg)"

    def __repr__(self) -> str:
        return self.__str__()


camera_matrices = [
    np.array(
        [
            [913.7377659, 0.0, 673.42504633],
            [0.0, 909.01346161, 464.51229778],
            [0.0, 0.0, 1.0],
        ]
    )
]
dist_coeffs = [
    np.array([[0.03171734, -0.01147495, -0.00010437, -0.00082573, -0.059311]])
]


def CALCULATE_PARTIAL_SOLUTION(
    camera_id: int, image: MatLike, all_corners, all_IDs
) -> List[Detection]:
    global CAM_FOV_X, CAM_FOV_Y, CAM_H

    h, w = image.shape[:2]

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrices[camera_id - 1],
        dist_coeffs[camera_id - 1],
        (w, h),
        1,
        (w, h),
    )

    result = []

    if all_IDs is None:
        return result

    for corners, tID in zip(all_corners, all_IDs):
        if str(tID) not in loaded_tags.keys():
            logger.Warn(f"Tag {tID} not found in loaded tags")
            continue

        corners = corners.flatten()
        x_br: float = corners[2]
        y_br: float = corners[3]

        tx_cm, ty_cm = GET_CAMERA_ANGLES(
            x_br,
            y_br,
            image,
            dist_coeffs[camera_id - 1][0],
            camera_matrices[camera_id - 1],
        )
        tx_cm += CAM_ANGLE_H.valueFloat()
        ty_cm -= CAM_ANGLE_V.valueFloat()

        tag_data = loaded_tags.get(str(tID), {})

        h_tag = float(tag_data.get("h", "54.0"))

        if ty_cm != 0:
            r_ground: float = (
                (CAM_H.valueFloat() - h_tag) / math.tan(math.radians(ty_cm))
            ) / math.cos(math.radians(tx_cm - CAM_ANGLE_H.valueFloat()))
            result.append(Detection(r_ground, tx_cm, tID))

    return result
