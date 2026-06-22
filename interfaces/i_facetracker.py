from abc import ABC, abstractmethod
import numpy as np

class IFaceTracker(ABC):
    @abstractmethod
    def start_track(self, frame: np.ndarray, bbox: tuple) -> None:
        """Инициализирует трекер на конкретном кадре по заданным координатам (x, y, w, h)"""
        pass

    @abstractmethod
    def update_track(self, frame: np.ndarray) -> tuple[bool, tuple]:
        """Обновляет позицию объекта на новом кадре. Возвращает (успех, (x, y, w, h))"""
        pass
