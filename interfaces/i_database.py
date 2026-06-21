from abc import ABC, abstractmethod
import numpy as np

class IFaceDatabase(ABC):
    """Интерфейс для работы с постоянным хранилищем векторов лиц."""
    @abstractmethod
    def load_all(self) -> tuple[list[np.ndarray], list[str]]:
        pass

    @abstractmethod
    def save(self, name: str, encoding: np.ndarray) -> None:
        pass
