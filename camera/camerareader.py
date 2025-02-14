
import cv2
from cv2.typing import MatLike
# from util.config import ConfigCategory, Config
from camera.preprocess import PROCESS_FRAME
from util.logger import Logger
from time import time_ns, sleep
from typing import Tuple
import numpy as np

logger = Logger("Camera")

camera_matrices = [np.array([[913.7377659,    0.,         673.42504633],
 [  0.,         909.01346161, 464.51229778],
 [  0.,           0.,           1.        ]])]
dist_coeffs = [np.array([[ 0.03171734, -0.01147495, -0.00010437, -0.00082573, -0.059311  ]])]

class CameraReader:
    def __init__(self, camera_id: int = 0):
        self.camera_id=(int)(camera_id)
        self.cap = cv2.VideoCapture(f"/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._{camera_id}_{camera_id}-video-index0")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, 120.0)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.loop_count=0

    def get_frame(self) -> Tuple[MatLike, int]:
        ret, frame = self.cap.read()
        while not ret:

            logger.Warn("Retrying get camera frame...")
            if (self.loop_count==0):
                self.cap = cv2.VideoCapture(f"/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._ATCam{self.camera_id}_ATCam{self.camera_id}-video-index0")
            
            self.loop_count=(self.loop_count+1)%100
            # sleep(0.1)
            ret, frame = self.cap.read()
        ts = time_ns()
        h, w = frame.shape[:2]

        
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrices[self.camera_id-1], dist_coeffs[self.camera_id-1], (w, h), 1, (w, h))
        frame = cv2.undistort(frame, camera_matrices[self.camera_id-1], dist_coeffs[self.camera_id-1], None, new_camera_matrix)

        frame = PROCESS_FRAME(frame)

        return frame, ts