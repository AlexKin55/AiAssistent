import cv2
import time
import numpy as np
from deepface import DeepFace
from interfaces import IFaceApp, IFaceDetector, IUserManager, IRenderer, IConversation, ISpeechSynthesizer

class PhotoFaceApp(IFaceApp):
    def __init__(self, detector: IFaceDetector, user_manager: IUserManager, renderer: IRenderer, conversation: IConversation, synthesizer: ISpeechSynthesizer):
        self.detector = detector
        self.user_manager = user_manager
        self.renderer = renderer
        self.conversation = conversation
        self.synthesizer = synthesizer
        
        self.current_user_name = "Scanning..."
        
        self.active_session_user = None
        self.last_seen_timestamp = time.time()
        self.session_timeout = 600.0

        self.pending_new_user = None
        self.new_user_strike_count = 0
        self.required_strikes = 5

    def _trigger_greeting_event(self, user_id: str):
        """Вызывается строго один раз при старте новой подтвержденной сессии визита"""
        print(f"\n[Контроль Сессий] Сессия ПОДТВЕРЖДЕНА для {user_id}. Генерирую приветствие...")
        
        trigger_phrase = "Клиент подошел к стойке и смотрит на тебя. Поприветствуй его согласно контексту."
        prompt_messages = self.conversation.build_prompt_messages(user_id, trigger_phrase)
        
        interests, _ = self.user_manager.get_llm_context(user_id)
        if interests:
            ai_response = f"Рада вас снова видеть! В прошлый раз вы интересовались товаром {interests[-1]}. Скажите, он ещё актуален?"
        else:
            ai_response = "Здравствуйте! Добро пожаловать в наш магазин. Чем я могу вам помочь?"
        
        print(f"🤖 ИИ-Продавец: {ai_response}")
        self.synthesizer.speak(ai_response)
        
        self.user_manager.add_message_to_history(user_id, "assistant", ai_response)
        self.last_seen_timestamp = time.time()
        print("[Контроль Сессий] Время сессии скорректировано после озвучки.")

    def run(self, camera_index: int = 0) -> None:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"Ошибка: Не удалось открыть камеру {camera_index}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print(f"\n=== Запущено PhotoFaceApp ===")
        print(f"[Настройка] Защита от дребезга лиц: {self.required_strikes} сек.")
        
        try:
            while cap.isOpened():
                for _ in range(5):
                    cap.grab()
                
                ret, frame = cap.retrieve()
                if not ret:
                    break

                top, right, bottom, left = 0, 0, 0, 0
                should_render = False
                detected_user_this_frame = None
                current_time = time.time()

                bboxes, face_crops = self.detector.detect_and_encode(frame)
                
                if bboxes and face_crops:
                    bbox = bboxes[0]
                    face_crop = face_crops[0]
                    
                    try:
                        resp = DeepFace.represent(
                            img_path=face_crop, 
                            model_name="Facenet512", 
                            detector_backend="skip",
                            enforce_detection=False
                        )
                        if resp:
                            embedding = resp[0]["embedding"] if isinstance(resp, list) else resp["embedding"]
                            self.current_user_name = self.user_manager.identify_or_create(embedding)
                            
                            detected_user_this_frame = self.current_user_name
                            
                            x, y, w, h = bbox
                            top, right, bottom, left = y, x + w, y + h, x
                            should_render = True
                    except Exception:
                        self.current_user_name = "Scanning..."
                else:
                    self.current_user_name = "No Face"

                if detected_user_this_frame is not None:
                    self.last_seen_timestamp = current_time

                    if self.active_session_user is None:
                        self.active_session_user = detected_user_this_frame
                        self.pending_new_user = None
                        self.new_user_strike_count = 0
                        self._trigger_greeting_event(detected_user_this_frame)
                        
                    elif self.active_session_user == detected_user_this_frame:
                        self.pending_new_user = None
                        self.new_user_strike_count = 0
                        
                    else:
                        if detected_user_this_frame != self.pending_new_user:
                            self.pending_new_user = detected_user_this_frame
                            self.new_user_strike_count = 1
                            print(f"[Контроль Дребезга] Замечен потенциальный новый ID: {detected_user_this_frame}. Проверка стабильности (1/{self.required_strikes})...")
                        else:
                            self.new_user_strike_count += 1
                            print(f"[Контроль Дребезга] ID {detected_user_this_frame} стабилен ({self.new_user_strike_count}/{self.required_strikes})")
                            
                            if self.new_user_strike_count >= self.required_strikes:
                                print(f"\n[Контроль Сессий] Смена подтверждена! Сессия {self.active_session_user} закрыта. Новый клиент: {detected_user_this_frame}.")
                                self.active_session_user = detected_user_this_frame
                                self.pending_new_user = None
                                self.new_user_strike_count = 0
                                self._trigger_greeting_event(detected_user_this_frame)
                else:
                    self.pending_new_user = None
                    self.new_user_strike_count = 0

                    if self.active_session_user is not None:
                        time_away = current_time - self.last_seen_timestamp
                        if time_away >= self.session_timeout:
                            print(f"\n[Контроль Сессий] Превышен лимит отсутствия ({int(time_away)} сек). Сессия для {self.active_session_user} закрыта.")
                            self.active_session_user = None

                if should_render:
                    self.renderer.render(frame, self.current_user_name, top, right, bottom, left)
                else:
                    cv2.putText(frame, "No Face Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                session_status = f"Active Session: {self.active_session_user}" if self.active_session_user else "Session: Closed"
                cv2.putText(frame, session_status, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if not self.renderer.show_and_check_exit(frame):
                    break

                time.sleep(1.0)
        finally:
            cap.release()
            self.renderer.close()
