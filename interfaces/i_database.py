from abc import ABC, abstractmethod
import numpy as np

class IFaceDatabase(ABC):
    @abstractmethod
    def _init_db(self):
        pass

    @abstractmethod
    def load_all_faces(self) -> dict[str, list[list[float]]]:
        pass

    @abstractmethod
    def save_or_update_face(self, user_id: str, embedding: list[float], purchases: str = "[]", history: str = "[]") -> None:
        pass

    @abstractmethod
    def get_user_data(self, user_id: str) -> tuple[str, str]:
        pass

    @abstractmethod
    def update_dialogue_history(self, user_id: str, history_json: str) -> None:
        pass

    @abstractmethod
    def update_timestamp_only(self, user_id: str) -> None:
        pass