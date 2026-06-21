# face_app.py
import cv2
import time
from interfaces import IFaceDetector, IUserManager, IRenderer

class FaceApp:
    def __init__(self, detector: IFaceDetector, user_manager: IUserManager, renderer: IRenderer):
        self.detector = detector
        self.user_manager = user_manager
        self.renderer = renderer
        self.cap = None

    def run(self, camera_index: int = 0) -> None:
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        
        if not self.cap.isOpened():
            print(f"[Ошибка] Нет доступа к камере")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("[Система] Инициализация камеры...")
        time.sleep(1.5)

        print("\n=== Система Face ID запущена ===")

        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue

            # 1. Поиск лиц (Оригинальный OpenCV-поиск в detector.py работает мгновенно)
            locations, encodings = self.detector.detect_and_encode(frame)

            # 2. Идентификация (Теперь этот метод асинхронный и отрабатывает за 0.001 секунды)
            for (top, right, bottom, left), encoding in zip(locations, encodings):
                name = self.user_manager.identify_or_register(encoding)
                self.renderer.render(frame, name, top, right, bottom, left)

            # 3. Вывод графики на экран
            if self.renderer.show_and_check_exit(frame):
                break

        self.cleanup()

    def cleanup(self) -> None:
        if self.cap:
            self.cap.release()
        self.renderer.close()
        print("=== Работа Face ID завершена ===")
