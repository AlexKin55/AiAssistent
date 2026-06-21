# database.py
import sqlite3
import numpy as np
from interfaces import IFaceDatabase

class SQLiteFaceDatabase(IFaceDatabase):

    def __init__(self, db_path: str = "faces_bio.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    face_embedding BLOB NOT NULL
                )
            """)
            conn.commit()

    def load_all(self) -> tuple[list[np.ndarray], list[str]]:
        encodings, names = [], []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, face_embedding FROM users")
            for name, blob in cursor.fetchall():
                encodings.append(np.frombuffer(blob, dtype=np.float64))
                names.append(name)
        return encodings, names

    def save(self, name: str, encoding: np.ndarray) -> None:
        blob_data = encoding.tobytes()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (name, face_embedding) VALUES (?, ?)", (name, blob_data))
            conn.commit()
