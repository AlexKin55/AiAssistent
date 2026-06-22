from interfaces import IUserManager
from scipy.spatial import distance

class DynamicUserManager(IUserManager):
    def __init__(self, threshold: float = 0.55, max_templates_per_user: int = 5):
        self.known_faces = {}  
        self.next_user_id = 1
        self.threshold = threshold
        self.max_templates_per_user = max_templates_per_user

    def identify_or_create(self, embedding: list[float]) -> str:
        best_match_id = None
        min_dist = 1.0

        for user_id, embeddings in self.known_faces.items():
            for known_emb in embeddings:
                dist = distance.cosine(embedding, known_emb)
                if dist < min_dist:
                    min_dist = dist
                    best_match_id = user_id

        if min_dist < self.threshold:
            if len(self.known_faces[best_match_id]) < self.max_templates_per_user:
                self.known_faces[best_match_id].append(embedding)
            
            print(f"[INFO] A Face was recognized: {best_match_id}")
            return f"User_{best_match_id}"
        else:
            self.known_faces[self.next_user_id] = [embedding]
            assigned_name = f"User_{self.next_user_id}"
            
            print(f"[INFO] A new Face was registered: {assigned_name}")
            self.next_user_id += 1
            
            return assigned_name