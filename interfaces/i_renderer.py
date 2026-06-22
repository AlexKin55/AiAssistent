from abc import ABC, abstractmethod
import numpy as np

class IRenderer(ABC):
    @abstractmethod
    def render(self, frame: np.ndarray, name: str, top: int, right: int, bottom: int, left: int) -> None:
        pass

    @abstractmethod
    def show_and_check_exit(self, frame: np.ndarray) -> bool:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
