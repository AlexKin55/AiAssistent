from abc import ABC, abstractmethod
import numpy as np

class IFaceDetector(ABC):
    """Интерфейс для обнаружения и извлечения эмбеддингов лиц."""
    @abstractmethod
    def detect_and_encode(self, frame: np.ndarray) -> tuple[list[tuple], list[np.ndarray]]:
        pass
