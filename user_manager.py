import numpy as np
from interfaces import IUserManager, IFaceDatabase


class AutoUserManager(IUserManager):
    """
    Реализация менеджера пользователей, который автоматически распознает 
    людей или регистрирует их "на лету" под именами User_N.
    """

    def __init__(self, db: IFaceDatabase, tolerance: float = 0.3):
        self.db = db
        self.tolerance = tolerance
        
        self.known_encodings, self.known_names = self.db.load_all()
        self._next_id = len(self.known_names) + 1

    def identify_or_register(self, face_encoding: np.ndarray) -> str:
        """
        Проверяет лицо по кэшу. Возвращает имя или создает новый профиль.
        """
        if len(self.known_encodings) > 0:
            distances = []
            
            for known_encoding in self.known_encodings:
                dist = np.linalg.norm(face_encoding - known_encoding)
                distances.append(dist)
            
            if len(distances) > 0:
                best_match_index = np.argmin(distances)
                
                if distances[best_match_index] <= self.tolerance:
                    return self.known_names[best_match_index]

        new_name = f"User_{self._next_id}"

        self._next_id += 1
        self.known_encodings.append(face_encoding)
        self.known_names.append(new_name)
        self.db.save(new_name, face_encoding)
        
        print(f"[UserManager] Запомнил новое лицо как: {new_name}")
        return new_name
