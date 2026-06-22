from abc import ABC, abstractmethod

class IConversation(ABC):

    @abstractmethod
    def build_prompt_messages(self, user_id: str, current_user_message: str) -> list[dict]:
        """
        Собирает идеальную структуру промпта (system + history + current) 
        для отправки в языковую модель (LLM).
        """
        pass
