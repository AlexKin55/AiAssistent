import time
import json
from scipy.spatial import distance
from interfaces import IUserManager, IFaceDatabase

class DynamicUserManager(IUserManager):
    def __init__(self, db: IFaceDatabase, threshold: float = 0.55, max_templates_per_user: int = 5):
        self.db = db
        self.threshold = threshold
        self.max_templates_per_user = max_templates_per_user
        self.known_faces = self.db.load_all_faces()
        self.last_db_update_time = {}
        self.db_update_interval_sec = 30.0
        self.max_history_len = 6  

        if self.known_faces:
            existing_ids = [int(uid.split('_')[1]) for uid in self.known_faces.keys() if uid.startswith('User_')]
            self.next_user_id = max(existing_ids) + 1 if existing_ids else 1
        else:
            self.next_user_id = 1

    def identify_or_create(self, embedding: list[float]) -> str:
        best_match_id = None
        min_dist = 1.0

        for user_id, embeddings in self.known_faces.items():
            for known_emb in embeddings:
                dist = distance.cosine(embedding, known_emb)
                if dist < min_dist:
                    min_dist = dist
                    best_match_id = user_id

        current_time = time.time()
        
        if min_dist < self.threshold:
            if len(self.known_faces[best_match_id]) < self.max_templates_per_user:
                self.known_faces[best_match_id].append(embedding)
            
            if current_time - self.last_db_update_time.get(best_match_id, 0.0) >= self.db_update_interval_sec:
                self.db.update_timestamp_only(best_match_id)
                self.last_db_update_time[best_match_id] = current_time
                print(f"[Менеджер] Обновлено время присутствия для {best_match_id}")
            return best_match_id

        elif self.threshold <= min_dist <= 0.68:
            if len(self.known_faces[best_match_id]) < self.max_templates_per_user:
                self.known_faces[best_match_id].append(embedding)
                
                raw_interests, raw_history = self.db.get_user_data(best_match_id)
                
                self.db.save_or_update_face(best_match_id, embedding, purchases=raw_interests, history=raw_history)
                self.last_db_update_time[best_match_id] = current_time
                print(f"[Менеджер] Успешно добавлен новый ракурс лица для {best_match_id}")
            else:
                if current_time - self.last_db_update_time.get(best_match_id, 0.0) >= self.db_update_interval_sec:
                    self.db.update_timestamp_only(best_match_id)
                    self.last_db_update_time[best_match_id] = current_time
            return best_match_id

        else:
            assigned_name = f"User_{self.next_user_id}"
            self.known_faces[assigned_name] = [embedding]
            
            self.db.save_or_update_face(assigned_name, embedding, purchases="[]", history="[]")
            self.last_db_update_time[assigned_name] = current_time
            
            print(f"[Менеджер] Зарегистрирован новый объект: {assigned_name}")
            self.next_user_id += 1
            return assigned_name

    def get_llm_context(self, user_id: str) -> tuple[list[str], list[dict]]:
        raw_interests, raw_history = self.db.get_user_data(user_id)
        try:
            interests = json.loads(raw_interests)
            history = json.loads(raw_history)
        except Exception:
            interests, history = [], []
        return interests, history

    def add_message_to_history(self, user_id: str, role: str, text: str) -> None:
        interests_list, chat_history = self.get_llm_context(user_id)
        chat_history.append({"role": role, "text": text})

        if len(chat_history) > self.max_history_len:
            chat_history.pop(0)
            
        self.db.update_dialogue_history(user_id, json.dumps(chat_history))
        self.last_db_update_time[user_id] = time.time()

    def add_user_interest(self, user_id: str, product_name: str) -> None:
        if user_id not in self.known_faces:
            return

        interests_list, _ = self.get_llm_context(user_id)
        
        if product_name not in interests_list:
            interests_list.append(product_name)
            if len(interests_list) > 5:
                interests_list.pop(0)
            
            embedding = self.known_faces[user_id][0]
            _, raw_history = self.db.get_user_data(user_id)
            
            self.db.save_or_update_face(
                user_id=user_id,
                embedding=embedding,
                purchases=json.dumps(interests_list),
                history=raw_history
            )
            print(f"[Менеджер] Товар '{product_name}' добавлен в интересы клиента {user_id}")
