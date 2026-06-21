# main.py
from database import SQLiteFaceDatabase
from detector import DeepFaceDetector
from user_manager import AutoUserManager
from renderer import OpenCvRenderer  # Добавили импорт
from face_app import FaceApp

def main():
    database = SQLiteFaceDatabase("faces_bio.db")
    detector = DeepFaceDetector(resize_factor=0.25)
    user_manager = AutoUserManager(database, tolerance=0.5)
    renderer = OpenCvRenderer(window_name="Face ID System")

    app = FaceApp(detector=detector, user_manager=user_manager, renderer=renderer)
    
    app.run(camera_index=0)

if __name__ == "__main__":
    main()
