from abc import ABC, abstractmethod
import numpy as np

class IAudioProcessor(ABC):
    @abstractmethod
    def listen_and_transcribe(self) -> str:
        pass
    
