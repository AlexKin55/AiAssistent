from abc import ABC, abstractmethod
import numpy as np

class IUserManager(ABC):
    @abstractmethod
    def identify_or_create(self, embedding: list[float]) -> str:
        pass

