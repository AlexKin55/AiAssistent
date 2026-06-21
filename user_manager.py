# user_manager.py
import numpy as np
# Импортируем интерфейсы программы
from interfaces import IUserManager, IFaceDatabase


class AutoUserManager(IUserManager):
    """
    Реализация менеджера пользователей, который автоматически распознает 
    людей или регистрирует их "на лету" под именами User_N.
    """

    def __init__(self, db: IFaceDatabase, tolerance: float = 0.3):
        """
        Args:
            db (IFaceDatabase): Любая база данных, реализующая контракт интерфейса.
            tolerance (float): Порог строгости сравнения для Евклидова расстояния.
                               Значение 0.3–0.4 оптимально для векторов DeepFace.
        """
        self.db = db
        self.tolerance = tolerance
        
        # Загружаем базу из SQLite в оперативную память (кэш)
        self.known_encodings, self.known_names = self.db.load_all()
        self._next_id = len(self.known_names) + 1

    def identify_or_register(self, face_encoding: np.ndarray) -> str:
        """
        Проверяет лицо по кэшу. Возвращает имя или создает новый профиль.
        """
        if len(self.known_encodings) > 0:
            distances = []
            
            # Считаем Евклидово расстояние (L2) между вектором с камеры и базой данных
            # Чистая математика через NumPy работает на уровне C и быстрее любых внешних библиотек
            for known_encoding in self.known_encodings:
                dist = np.linalg.norm(face_encoding - known_encoding)
                distances.append(dist)
            
            if len(distances) > 0:
                # Находим индекс самого похожего человека (минимальное расстояние)
                best_match_index = np.argmin(distances)
                
                # Если расстояние укладывается в порог — человек успешно распознан
                if distances[best_match_index] <= self.tolerance:
                    return self.known_names[best_match_index]

        # АВТОЗАПОМИНАНИЕ: Если лицо новое
        new_name = f"User_{self._next_id}"
        self._next_id += 1
        
        # Обновляем кэш в оперативной памяти
        self.known_encodings.append(face_encoding)
        self.known_names.append(new_name)
        
        # Сохраняем в физическую БД SQLite через интерфейс
        self.db.save(new_name, face_encoding)
        
        print(f"[UserManager] Запомнил новое лицо как: {new_name}")
        return new_name
