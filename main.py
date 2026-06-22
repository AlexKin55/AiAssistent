from detector import MediaPipeLightDetector
from facetracker import OpenCVKCFTracker
from renderer import OpenCVRenderer
from user_manager import DynamicUserManager
from database import SQLiteFaceDatabase
from conversation import RetailAssistantConversation
from yspeech_synthesizer import YandexSpeechKitSynthesizer
from yaudio_processor import YandexSpeechKitProcessor 
from y_llm_client import YandexGPTLLMClient

from video_app import VideoFaceApp
from photo_app import PhotoFaceApp


YANDEX_API_KEY = ""
YANDEX_FOLDER_ID = ""

def main() -> None:

    detector_impl = MediaPipeLightDetector(min_detection_confidence=0.82)
    db_impl = SQLiteFaceDatabase(db_name="face_intelligence.db")
    tracker_impl = OpenCVKCFTracker()
    renderer_impl = OpenCVRenderer(window_name="Smart Face Pipeline")
    user_manager_impl = DynamicUserManager(db = db_impl, threshold=0.55, max_templates_per_user=5)
    conversation_impl = RetailAssistantConversation(user_manager_impl)
    synthesizer_impl = YandexSpeechKitSynthesizer(api_key=YANDEX_API_KEY, folder_id=YANDEX_FOLDER_ID)
    audio_processor_impl = YandexSpeechKitProcessor(api_key=YANDEX_API_KEY, folder_id=YANDEX_FOLDER_ID)
    llm_client_impl = YandexGPTLLMClient(api_key=YANDEX_API_KEY, folder_id=YANDEX_FOLDER_ID)
    # Режим А: Видео 30 FPS с трекером (Нагрузка на CPU выше)
    # app = VideoFaceApp(detector_impl, user_manager_impl, renderer_impl, tracker_impl)
    
    app = PhotoFaceApp(detector_impl, user_manager_impl, renderer_impl, conversation_impl,
                       synthesizer_impl, audio_processor_impl, llm_client_impl)

    app.run(camera_index=0)

if __name__ == "__main__":
    main()
