# detector.py
import cv2
import numpy as np
from deepface import DeepFace
from interfaces import IFaceDetector

class DeepFaceDetector(IFaceDetector):
    """Реализация детектора на базе современной библиотеки DeepFace."""

    def __init__(self, resize_factor: float = 0.25):
        """
        Инициализация детектора.
        
        Args:
            resize_factor (float): Коэффициент сжатия кадра (0.25 = в 4 раза).
        """
        self.resize_factor = resize_factor

    def detect_and_encode(self, frame: np.ndarray) -> tuple[list[tuple], list[np.ndarray]]:
        if frame is None or frame.size == 0:
            return [], []
        
        try:
            # DeepFace находит лица и извлекает векторы с помощью модели VGG-Face
            # enforce_detection=False предотвращает падение кода, если лиц в кадре нет
            predictions = DeepFace.represent(img_path=frame, model_name="VGG-Face", enforce_detection=False)
            
            locations = []
            encodings = []
            
            for pred in predictions:
                # Извлекаем координаты лица (словарь x, y, w, h)
                box = pred["facial_area"]
                
                # Переводим в формат top, right, bottom, left для нашего FaceApp
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
