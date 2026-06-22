import cv2
import time
import numpy as np
from deepface import DeepFace
from interfaces import IFaceApp, IFaceDetector, IUserManager, IRenderer

class PhotoFaceApp(IFaceApp):
    def __init__(self, detector: IFaceDetector, user_manager: IUserManager, renderer: IRenderer):
        self.detector = detector
        self.user_manager = user_manager
        self.renderer = renderer
        self.current_user_name = "Scanning..."

    def run(self, camera_index: int = 0) -> None:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"Ошибка: Не удалось открыть камеру {camera_index}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("\n=== Запущено PhotoFaceApp (Эко-режим: 1 фото/сек) ===")
        try:
            while cap.isOpened():
                # Очищаем буфер старых кадров веб-камеры
                for _ in range(5):
                    cap.grab()
                
                ret, frame = cap.retrieve()
                if not ret:
                    break

                top, right, bottom, left = 0, 0, 0, 0
                should_render = False

                # Прямая детекция
                bboxes, face_crops = self.detector.detect_and_encode(frame)
                
                if bboxes and face_crops:
                    bbox = bboxes[0]
                    face_crop = face_crops[0]
                    
                    try:
                        resp = DeepFace.represent(
                            img_path=face_crop, 
                            model_name="Facenet512", 
                            detector_backend="skip",
                            enforce_detection=False
                        )
                        if resp:
                            embedding = resp[0]["embedding"] if isinstance(resp, list) else resp["embedding"]
                            self.current_user_name = self.user_manager.identify_or_create(embedding)
                            
                            x, y, w, h = bbox
                            top, right, bottom, left = y, x + w, y + h, x
                            should_render = True
                    except Exception:
                        self.current_user_name = "Scanning..."
                else:
                    self.current_user_name = "No Face"

                if should_render:
                    self.renderer.render(frame, self.current_user_name, top, right, bottom, left)
                else:
                    cv2.putText(frame, "No Face Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.putText(frame, "Mode: Eco Photo (1 FPS)", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if not self.renderer.show_and_check_exit(frame):
                    break

                # Отдых для CPU
                time.sleep(1.0)
        finally:
            cap.release()
            self.renderer.close()
