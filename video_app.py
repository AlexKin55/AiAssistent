import cv2
import numpy as np
from deepface import DeepFace

from interfaces import IFaceDetector
from interfaces import IUserManager
from interfaces import IRenderer
from interfaces import IFaceTracker

class VideoFaceApp:
    def __init__(self, detector: IFaceDetector, user_manager: IUserManager, renderer: IRenderer, tracker: IFaceTracker):
        self.detector = detector
        self.user_manager = user_manager
        self.renderer = renderer
        self.tracker = tracker
        self.tracking_active = False
        self.current_user_name = "Scanning..."
        
        self.track_refresh_frames = 30
        self.frame_count = 0

    def _recognize_and_start_track(self, frame: np.ndarray) -> bool:
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
                    self.tracker.start_track(frame, bbox)
                    self.tracking_active = True
                    return True
                    
            except Exception as e:
                print(f"[FATAL ERROR DEEPFACE]: {e}")
                self.tracking_active = False
                return False
                
        self.tracking_active = False
        return False

    def run(self, camera_index: int = 0) -> None:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print(f"ERROR: Couldn't get access to the camera by index {camera_index}")
            return

        print("\n=== FaceApp was started ===")
        print("Press 'q' or 'Esc' to exit the app.\n")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print(f"[WARNING] A frame wasnt received from the camera {camera_index}. Try to another camera index.")
                    break
                    
                self.frame_count += 1
                top, right, bottom, left = 0, 0, 0, 0
                should_render = False

                if not self.tracking_active or (self.frame_count % self.track_refresh_frames == 0):
                    success = self._recognize_and_start_track(frame)
                    if success:
                        success_track, bbox = self.tracker.update_track(frame)
                        if success_track:
                            x, y, w, h = bbox
                            top, right, bottom, left = y, x + w, y + h, x
                            should_render = True
                else:
                    success_track, bbox = self.tracker.update_track(frame)
                    if success_track:
                        x, y, w, h = bbox
                        top, right, bottom, left = y, x + w, y + h, x
                        should_render = True
                    else:
                        self.tracking_active = False
                        self.current_user_name = "Scanning..."

                if should_render:
                    self.renderer.render(frame, self.current_user_name, top, right, bottom, left)
                else:
                    cv2.putText(frame, "No Face Detected", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                mode_text = "Mode: Fast Tracking" if self.tracking_active else "Mode: DeepFace Inference"
                cv2.putText(frame, mode_text, (20, frame.shape[0] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

                if not self.renderer.show_and_check_exit(frame):
                    break
                    
        finally:
            cap.release()
            self.renderer.close()
            print("[INFO] Session was closed successfully.")

