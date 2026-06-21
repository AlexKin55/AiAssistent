# detector.py
import cv2
import numpy as np
from deepface import DeepFace
from interfaces import IFaceDetector

class DeepFaceDetector(IFaceDetector):
    """Сверхбыстрый детектор, разделяющий OpenCV-поиск и DeepFace-анализ."""

    def __init__(self, resize_factor: float = 0.25):
        self.resize_factor = resize_factor
        # Загружаем встроенный в OpenCV легкий C++ детектор лиц
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.model_name = "Facenet"

    def detect_and_encode(self, frame: np.ndarray) -> tuple[list[tuple], list[np.ndarray]]:
        if frame is None or frame.size == 0:
            return [], []

        # 1. МГНОВЕННЫЙ ПОИСК ЛИЦА ЧЕРЕЗ OPENCV (работает на С++, 0% тормозов)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        locations = []
        encodings = []

        # Если OpenCV вообще не нашел лиц в кадре, сразу выходим (не дергаем нейросеть)
        if len(faces) == 0:
            return [], []

        for (x, y, w, h) in faces:
            # Переводим координаты в формат top, right, bottom, left для интерфейса
            top, right, bottom, left = y, x + w, y + h, x
            locations.append((top, right, bottom, left))

            # 2. ОПТИМИЗИРОВАННЫЙ СЕМПЛИНГ НЕЙРОСЕТИ
            # Вырезаем область лица, чтобы нейросеть не обсчитывала весь кадр целиком
            face_img = frame[y:y+h, x:x+w]
            
            if face_img.size == 0:
                continue

            try:
                # Извлекаем эмбеддинг только из маленького кусочка лица
                # align=False отключает тяжелую процедуру выравнивания (главный тормоз CPU!)
                predictions = DeepFace.represent(
                    img_path=face_img,
                    model_name=self.model_name,
                    detector_backend="skip", # Пропускаем внутренний детектор DeepFace
                    align=False,              # Отключаем выравнивание для дикой скорости
                    enforce_detection=False
                )
                
                if predictions:
                    encodings.append(np.array(predictions[0]["embedding"]))
            except Exception:
                # Если нейросеть не смогла считать дескриптор, возвращаем пустой вектор
                encodings.append(np.zeros(128)) 

        return locations, encodings
