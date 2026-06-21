# face_app.py
import cv2
import time  # Добавляем стандартный модуль времени
from interfaces import IFaceDetector, IUserManager, IRenderer

class FaceApp:
    def __init__(self, detector: IFaceDetector, user_manager: IUserManager, renderer: IRenderer):
        self.detector = detector
        self.user_manager = user_manager
        self.renderer = renderer
        self.cap = None

    def run(self, camera_index: int = 0) -> None:
        # Указываем OpenCV использовать конкретный драйвер Linux - CAP_V4L2
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        
        if not self.cap.isOpened():
            print(f"[Ошибка] Нет доступа к камере с индексом {camera_index}")
            return

        # СТАБИЛИЗАЦИЯ: Явно задаем стандартное разрешение (снижает нагрузку на шину)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # СТАБИЛИЗАЦИЯ: Даем камере 1.5 секунды, чтобы проснуться и настроить экспозицию
        print("[Система] Инициализация камеры, пожалуйста, подождите...")
        time.sleep(1.5)

        print("\n=== Система Face ID запущена ===")

        while True:
            ret, frame = self.cap.read()
            
            # СТАБИЛИЗАЦИЯ: Если один кадр случайно пропустился, не закрываем программу сразу,
            # а пробуем прочитать следующий кадр в цикле.
            if not ret or frame is None:
                continue

            # Поиск лиц
            locations, encodings = self.detector.detect_and_encode(frame)

            # Идентификация и прорисовка
            for (top, right, bottom, left), encoding in zip(locations, encodings):
                name = self.user_manager.identify_or_register(encoding)
                self.renderer.render(frame, name, top, right, bottom, left)

            # Вывод на экран и проверка нажатия клавиши 'q'
            if self.renderer.show_and_check_exit(frame):
                break

        self.cleanup()

    def cleanup(self) -> None:
        if self.cap:
            self.cap.release()
        self.renderer.close()
        print("=== Работа Face ID успешно завершена ===")
