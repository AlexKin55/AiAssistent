# detector.py
import cv2
import numpy as np
from deepface import DeepFace
from interfaces import IFaceDetector

class DeepFaceDetector(IFaceDetector):
    """Реализация детектора на базе современной библиотеки DeepFace."""

    def __init__(self, resize_factor: float = 0.25):
        self.resize_factor = resize_factor

    def detect_and_encode(self, frame: np.ndarray) -> tuple[list[tuple], list[np.ndarray]]:
        if frame is None or frame.size == 0:
            return [], []
        
        try:
            predictions = DeepFace.represent(img_path=frame, model_name="VGG-Face", enforce_detection=False)
            
            locations = []
            encodings = []
            
            for pred in predictions:
                box = pred["facial_area"]
                
                top = box["y"]
                right = box["x"] + box["w"]
                bottom = box["y"] + box["h"]
                left = box["x"]
                
                locations.append((top, right, bottom, left))
                encodings.append(np.array(pred["embedding"]))
                
            return locations, encodings
        except Exception as e:
            print(f"[Detector Error] {e}")
            return [], []
