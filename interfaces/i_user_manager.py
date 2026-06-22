from abc import ABC, abstractmethod
import numpy as np

class IUserManager(ABC):
    @abstractmethod
    def identify_or_create(self, embedding: list[float]) -> str:
        pass
    
    @abstractmethod
    def get_llm_context(self, user_id: str) -> tuple[list[str], list[dict]]:
        pass

    @abstractmethod
    def add_message_to_history(self, user_id: str, role: str, text: str) -> None:
        pass