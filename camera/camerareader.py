
import cv2
from cv2.typing import MatLike
from util.config import ConfigCategory, Config
from camera.preprocess import PROCESS_FRAME
from util.logger import Logger
from time import time_ns, sleep
from typing import Tuple

logger = Logger("Camera")

class CameraReader:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, 120.0)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)

        self.og_frame = None

    def get_frame(self) -> Tuple[MatLike, int]:
        ret, frame = self.cap.read()
        self.og_frame = frame

        while not ret:
            logger.Warn("Retrying get camera frame...")
            sleep(0.1)
            ret, frame = self.cap.read()

        ts = time_ns()
        frame = PROCESS_FRAME(frame)

        return frame, ts
    
    def get_og_frame(self) -> MatLike:
        self.og_frame = cv2.cvtColor(self.og_frame, cv2.COLOR_BGR2GRAY)
        return self.og_frame