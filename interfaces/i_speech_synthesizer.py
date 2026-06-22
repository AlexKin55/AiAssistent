from abc import ABC, abstractmethod

class ISpeechSynthesizer(ABC):

    @abstractmethod
    def speak(self, text: str) -> None:
        pass
