import os
import sqlite3
import json
from datetime import datetime
from interfaces import IFaceDatabase

class SQLiteFaceDatabase(IFaceDatabase):
    def __init__(self, db_name: str = "face_intelligence.db"):
        self.db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), db_name))
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    user_id TEXT PRIMARY KEY,
                    embedding TEXT NOT NULL,
                    last_seen TIMESTAMP NOT NULL,
                    purchase_history TEXT,
                    conversation_history TEXT
                )
            """)
            conn.commit()

    def load_all_faces(self) -> dict[str, list[list[float]]]:
        faces_data = {}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id, embedding FROM face_embeddings")
            rows = cursor.fetchall()
            for user_id, emb_json in rows:
                faces_data[user_id] = [json.loads(emb_json)]
        print(f"[БД SQLite] Успешно загружено профилей в ОЗУ: {len(faces_data)}")
        return faces_data

    def save_or_update_face(self, user_id: str, embedding: list[float], purchases: str = "[]", history: str = "[]") -> None:
        emb_json = json.dumps(embedding)
        current_time = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO face_embeddings (user_id, embedding, last_seen, purchase_history, conversation_history)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, emb_json, current_time, purchases, history))
            conn.commit()

    def get_user_data(self, user_id: str) -> tuple[str, str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT purchase_history, conversation_history FROM face_embeddings WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                interests_json = row[0] if row[0] is not None else "[]"
                history_json = row[1] if row[1] is not None else "[]"
                return interests_json, history_json
            return "[]", "[]"

    def update_dialogue_history(self, user_id: str, history_json: str) -> None:
        current_time = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE face_embeddings 
                SET last_seen = ?, conversation_history = ? 
                WHERE user_id = ?
            """, (current_time, history_json, user_id))
            conn.commit()

    def update_timestamp_only(self, user_id: str) -> None:
        current_time = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE face_embeddings SET last_seen = ? WHERE user_id = ?", (current_time, user_id))
            conn.commit()
