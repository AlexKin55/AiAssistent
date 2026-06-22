from interfaces import IRenderer

import cv2
import numpy as np

class OpenCVRenderer(IRenderer):
    def __init__(self, window_name: str = "Smart Face App"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def render(self, frame: np.ndarray, name: str, top: int, right: int, bottom: int, left: int) -> None:
        if frame is None or frame.size == 0:
            return

        box_color = (0, 255, 0)
        text_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        cv2.rectangle(frame, (left, top), (right, bottom), box_color, thickness)

        label_size, _ = cv2.getTextSize(name, font, font_scale, thickness)
        text_w, text_h = label_size[0], label_size[1]
        
        cv2.rectangle(
            frame, 
            (left, top - text_h - 12), 
            (left + text_w + 6, top), 
            box_color, 
            cv2.FILLED
        )

        cv2.putText(
            frame, 
            name, 
            (left + 3, top - 5), 
            font, 
            font_scale, 
            text_color, 
            thickness, 
            cv2.LINE_AA
        )

    def show_and_check_exit(self, frame: np.ndarray) -> bool:
        if frame is not None and frame.size > 0:
            cv2.imshow(self.window_name, frame)
            
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            return False
        return True

    def close(self) -> None:
        cv2.destroyAllWindows()
