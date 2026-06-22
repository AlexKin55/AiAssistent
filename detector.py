import os
import cv2
import numpy as np
from interfaces import IFaceDetector

class MediaPipeLightDetector(IFaceDetector):
    def __init__(self, min_detection_confidence: float = 0.82): # Повысили порог по умолчанию до 0.82
        model_name = "face_detection_yunet_2023mar.onnx"
        self.model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), model_name))
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"\n\n[КРИТИЧЕСКАЯ ОШИБКА]: Файл '{model_name}' отсутствует в папке проекта!\n"
                f"Пожалуйста, скачайте его и положите строго сюда: {self.model_path}\n"
            )

        self.min_confidence = min_detection_confidence
        
        self.detector = cv2.FaceDetectorYN.create(
            self.model_path,    # 1. model
            "",                 # 2. config
            (320, 320),         # 3. inputSize
            self.min_confidence,# 4. scoreThreshold
            0.3,                # 5. nmsThreshold
            5                   # 6. topK
        )
        print(f"[Отладка Детектора] Детектор YuNet инициализирован. Порог фокуса: {self.min_confidence}")

    def detect_and_encode(self, frame: np.ndarray) -> tuple[list[tuple], list[np.ndarray]]:
        bboxes = []
        face_crops = []
        
        if frame is None or frame.size == 0:
            return bboxes, face_crops

        h, w, _ = frame.shape
        self.detector.setInputSize((w, h))
        
        _, faces = self.detector.detect(frame)

        if faces is not None:
            for face in faces:
                confidence_score = float(face[14])
                
                x = int(face[0])
                y = int(face[1])
                width = int(face[2])
                height = int(face[3])
                min_allowed_height = int(h * 0.12) 
                
                if height < min_allowed_height:
                    continue

                x, y = max(0, x), max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)

                if width > 0 and height > 0:
                    bboxes.append((x, y, width, height))
                    face_crops.append(frame[y:y+height, x:x+width])

        return bboxes, face_crops
