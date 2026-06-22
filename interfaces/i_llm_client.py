from abc import ABC, abstractmethod

class ILLMClient(ABC):

    @abstractmethod
    def generate_response(self, prompt_messages: list[dict]) -> str:
        """
        Отправляет структурированный массив сообщений (system, user, assistant)
        в текстовую нейросеть и возвращает сгенерированный ИИ ответ.
        """
        pass
