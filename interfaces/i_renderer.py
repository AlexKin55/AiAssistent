# interfaces/i_renderer.py
from abc import ABC, abstractmethod
import numpy as np

class IRenderer(ABC):
    """Интерфейс для отрисовки результатов распознавания и вывода на экран."""
    
    @abstractmethod
    def render(self, frame: np.ndarray, name: str, top: int, right: int, bottom: int, left: int) -> None:
        """Рисует рамку и имя вокруг обнаруженного лица."""
        pass

    @abstractmethod
    def show_and_check_exit(self, frame: np.ndarray) -> bool:
        """Показывает финальный кадр в окне и проверяет нажатие клавиши выхода (True — выйти)."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Закрывает графические окна и освобождает ресурсы."""
        pass
