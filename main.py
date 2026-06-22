from detector import MediaPipeLightDetector
from facetracker import OpenCVKCFTracker
from renderer import OpenCVRenderer
from user_manager import DynamicUserManager
from database import SQLiteFaceDatabase
from conversation import RetailAssistantConversation

# Импортируем две разные стратегии работы приложения
from video_app import VideoFaceApp
from photo_app import PhotoFaceApp

def main() -> None:
    # 1. Сборка общих зависимостей
    detector_impl = MediaPipeLightDetector(min_detection_confidence=0.82)
    db_impl = SQLiteFaceDatabase(db_name="face_intelligence.db")
    tracker_impl = OpenCVKCFTracker()
    renderer_impl = OpenCVRenderer(window_name="Smart Face Pipeline")
    user_manager_impl = DynamicUserManager(db = db_impl, threshold=0.55, max_templates_per_user=5)
    conversation_impl = RetailAssistantConversation(user_manager_impl)
    # Режим А: Видео 30 FPS с трекером (Нагрузка на CPU выше)
    # app = VideoFaceApp(detector_impl, user_manager_impl, renderer_impl, tracker_impl)
    
    # Режим Б: Фото 1 FPS (Нагрузка на CPU минимальна, кулеры молчат)
    app = PhotoFaceApp(detector_impl, user_manager_impl, renderer_impl, conversation_impl)

    app.run(camera_index=0)

if __name__ == "__main__":
    main()
