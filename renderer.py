# renderer.py
import cv2
import numpy as np
from interfaces import IRenderer

class OpenCvRenderer(IRenderer):
    """Реализация вывода графики с помощью стандартных окон OpenCV."""

    def __init__(self, window_name: str = "Face ID System"):
        self.window_name = window_name

    def render(self, frame: np.ndarray, name: str, top: int, right: int, bottom: int, left: int) -> None:
        accent_color = (0, 255, 0)
        text_color = (255, 255, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), accent_color, 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), accent_color, cv2.FILLED)
        cv2.putText(
            frame, 
            name, 
            (left + 6, bottom - 6), 
            cv2.FONT_HERSHEY_DUPLEX, 
            0.7, 
            text_color, 
            1
        )

    def show_and_check_exit(self, frame: np.ndarray) -> bool:
        cv2.imshow(self.window_name, frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return True
        return False

    def close(self) -> None:
        cv2.destroyAllWindows()
