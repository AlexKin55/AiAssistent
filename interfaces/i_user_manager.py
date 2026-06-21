from abc import ABC, abstractmethod
import numpy as np

class IUserManager(ABC):
    """Интерфейс для управления пользователями и генерации ID."""
    @abstractmethod
    def identify_or_register(self, face_encoding: np.ndarray) -> str:
        pass
