from abc import ABC, abstractmethod

class IFaceApp(ABC):
    
    @abstractmethod
    def run(self, camera_index: int = 0) -> None:
        pass
