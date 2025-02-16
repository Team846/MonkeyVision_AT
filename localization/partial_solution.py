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


CAM_ANGLE_H = 0

TAG_H = 0


def SET_CAM(pipeline: int):
    pref_category = ConfigCategory(f"PartialSolution{pipeline}")
    global CAM_ANGLE_H, CAM_H, TAG_H
    CAM_ANGLE_H = pref_category.getFloatConfig("CAM_MOUNT_H_deg", 0.0)
    TAG_H = pref_category.getFloatConfig("TAG_H_in", 6.25)


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
    global CAM_ANGLE_H, TAG_H

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
        x_left: float = (corners[2] + corners[4]) / 2.0
        y_bottom: float = (corners[3] + corners[1]) / 2.0
        y_top: float = (corners[5] + corners[7]) / 2.0

        tx_l, ty_t = GET_CAMERA_ANGLES(
            x_left,
            y_top,
            image,
            dist_coeffs[camera_id - 1][0],
            new_camera_matrix,
        )
        _, ty_b = GET_CAMERA_ANGLES(
            0.0,
            y_bottom,
            image,
            dist_coeffs[camera_id - 1][0],
            new_camera_matrix,
        )
        tx_l += CAM_ANGLE_H.valueFloat()

        r_ground: float = TAG_H.valueFloat() / abs(
            math.tan(math.radians(ty_t)) - math.tan(math.radians(ty_b))
        )
        r_ground = r_ground / math.cos(math.radians(tx_l))

        result.append(Detection(r_ground, tx_l, tID))

    return result
