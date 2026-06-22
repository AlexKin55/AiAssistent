from abc import ABC, abstractmethod

class IConversation(ABC):

    @abstractmethod
    def build_prompt_messages(self, user_id: str, current_user_message: str) -> list[dict]:
        pass

    @abstractmethod
    def build_sql_prompt(self, user_id: str, current_user_message: str) -> list[dict]:
        pass
