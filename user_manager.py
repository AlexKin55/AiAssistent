# user_manager.py
import threading
import numpy as np
from deepface import DeepFace
from interfaces import IUserManager, IFaceDatabase


class AutoUserManager(IUserManager):
    """
    Асинхронный менеджер пользователей.
    Выносит тяжелые вычисления DeepFace в отдельный фоновый поток,
    чтобы видеопоток с камеры работал на максимальной скорости (30 FPS).
    """

    def __init__(self, db: IFaceDatabase, tolerance: float = 0.3):
        self.db = db
        self.tolerance = tolerance
        
        # Загружаем базу из SQLite в кэш
        self.known_encodings, self.known_names = self.db.load_all()
        self._next_id = len(self.known_names) + 1

        # Потокобезопасность и управление фоновым потоком
        self._is_processing = False  # Флаг: занята ли нейросеть вычислениями прямо сейчас
        self._last_detected_name = "Unknown"  # Последнее успешно распознанное имя
        self._lock = threading.Lock()

    def identify_or_register(self, face_encoding: np.ndarray) -> str:
        """
        Мгновенно возвращает последнее известное имя. 
        Если фоновый поток свободен, запускает в нем новый анализ лица.
        """
        # Если фоновый поток уже считает предыдущий кадр, не ждем его.
        # Мгновенно возвращаем старый результат, чтобы видео не тормозило.
        if self._is_processing:
            return self._last_detected_name

        # Если фоновый поток свободен, запускаем в нем тяжелый анализ лица
        self._is_processing = True
        
        # Передаем копию вектора лица в фоновый поток
        threading.Thread(
            target=self._background_analysis, 
            args=(face_encoding.copy(),), 
            daemon=True
        ).start()

        return self._last_detected_name

    def _background_analysis(self, face_encoding: np.ndarray):
        """Этот метод выполняется в отдельном потоке и не блокирует камеру."""
        try:
            name = "Unknown"

            if len(self.known_encodings) > 0:
                distances = []
                for known_encoding in self.known_encodings:
                    dist = np.linalg.norm(face_encoding - known_encoding)
                    distances.append(dist)
                
                if len(distances) > 0:
                    best_match_index = np.argmin(distances)
                    if distances[best_match_index] <= self.tolerance:
                        name = self.known_names[best_match_index]

            # Если лицо новое — регистрируем его (тоже в фоне)
            if name == "Unknown":
                with self._lock:
                    name = f"User_{self._next_id}"
                    self._next_id += 1
                    
                    self.known_encodings.append(face_encoding)
                    self.known_names.append(name)
                    self.db.save(name, face_encoding)
                print(f"[UserManager] В фоне зарегистрировано новое лицо: {name}")

            # Сохраняем результат для вывода на экран
            self._last_detected_name = name

        except Exception as e:
            print(f"[Фоновый поток] Ошибка анализа: {e}")
        finally:
            # Освобождаем поток для следующего кадра
            self._is_processing = False
