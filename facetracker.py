from interfaces import IFaceTracker

import cv2
import numpy as np

class OpenCVKCFTracker(IFaceTracker):
    def __init__(self):
        self.tracker = None

    def start_track(self, frame: np.ndarray, bbox: tuple) -> None:
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, bbox)

    def update_track(self, frame: np.ndarray) -> tuple[bool, tuple]:
        if self.tracker is None:
            return False, (0, 0, 0, 0)
            
        success, bbox = self.tracker.update(frame)
        if success:
            return True, tuple(int(v) for v in bbox)
            
        return False, (0, 0, 0, 0)