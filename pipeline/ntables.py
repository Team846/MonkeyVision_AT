import cv2
from threading import Thread
from typing import List
from ntcore import NetworkTableInstance

class NTables:
    def __init__(self, pipeline_number):
        self.inst = NetworkTableInstance.getDefault()
        self.inst.startClient4("AprilTagServer")
        self.inst.setServerTeam(846)
        self.inst.startDSClient()
        self.table = self.inst.getTable(f"AprilTagsCam{pipeline_number}")

    def execute(self, detections, latency):
        self.table.putNumber("tl", latency)
        angles:List[float]=[]
        distances:List[float]=[]
        tags:List[int]=[]
        for detection in detections:
            angles.append(detection.getTheta())
            distances.append(detection.getR())
            tags.append(detection.getTag())
        self.table.putNumberArray("tx", angles)
        self.table.putNumberArray("distances", distances)
        self.table.putNumberArray("tags", tags)
        pass

    def updateFrameNum(self, frame_num):
        self.table.putNumber("curFrameNum", frame_num)
        pass

