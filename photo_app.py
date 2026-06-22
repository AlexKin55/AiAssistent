import cv2
import time
import numpy as np
from deepface import DeepFace
from interfaces import IFaceApp, IFaceDetector, IUserManager, IRenderer, IConversation, ISpeechSynthesizer, IAudioProcessor, ILLMClient

class PhotoFaceApp(IFaceApp):
    def __init__(self, detector: IFaceDetector, user_manager: IUserManager, renderer: IRenderer,
                 conversation: IConversation, synthesizer: ISpeechSynthesizer, audio_processor: IAudioProcessor, llm_client: ILLMClient):
        self.detector = detector
        self.user_manager = user_manager
        self.renderer = renderer
        self.conversation = conversation
        self.synthesizer = synthesizer
        self.audio_processor = audio_processor
        self.llm_client = llm_client
        
        self.current_user_name = "Scanning..."
        
        self.active_session_user = None
        self.last_seen_timestamp = time.time()
        self.session_timeout = 600.0

        self.pending_new_user = None
        self.new_user_strike_count = 0
        self.required_strikes = 5

    def _trigger_greeting_event(self, user_id: str):
        """Срабатывает один раз при входе человека в кадр"""
        print(f"\n[Контроль Сессий] Сессия ПОДТВЕРЖДЕНА для {user_id}. ИИ генерирует приветствие...")
        
        trigger_phrase = "Клиент подошел к стойке и смотрит на тебя. Поприветствуй его согласно контексту."
        prompt_messages = self.conversation.build_prompt_messages(user_id, trigger_phrase)
        
        ai_response = self.llm_client.generate_response(prompt_messages)
        print(f"🤖 Алиса: {ai_response}")
        
        self.synthesizer.speak(ai_response)
        self.user_manager.add_message_to_history(user_id, "assistant", ai_response)
        self.last_seen_timestamp = time.time()


    def _process_voice_dialogue(self, user_id: str, user_text: str):
        """Двухэтапный ИИ-конвейер: обработка живой речи пользователя"""
        print(f"\n[Голосовой Конвейер] Вы сказали: '{user_text}'")

        sql_prompt = self.conversation.build_sql_prompt(user_text)
        generated_sql = self.llm_client.generate_response(sql_prompt).strip()
        
        db_result = []
        if generated_sql.upper().startswith("SELECT"):
            print(f"[ИИ Анализ] Сгенерирован SQL-запрос к складу: {generated_sql}")
            # Делаем прямой безопасный запрос к таблице продуктов через объект user_manager.db
            db_result = self.user_manager.db.execute_read_query(generated_sql)
            print(f"[БД SQLite] Результат из таблицы товаров: {db_result}")

        chat_prompt = self.conversation.build_prompt_messages(user_id, user_text)

        if db_result:
            chat_prompt[-1]["text"] += f"\n(СИСТЕМНАЯ СПРАВКА ИЗ SQLite БД: Результат выполнения SQL-запроса к складу товаров: {db_result})"

        ai_response = self.llm_client.generate_response(chat_prompt)
        print(f"🤖 Алиса: {ai_response}")

        self.synthesizer.speak(ai_response)
        self.user_manager.add_message_to_history(user_id, "user", user_text)
        self.user_manager.add_message_to_history(user_id, "assistant", ai_response)
        self.last_seen_timestamp = time.time()

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


                if self.active_session_user is not None:
                    recognized_text = self.audio_processor.listen_and_transcribe()
                    if recognized_text and recognized_text.strip():
                        self._process_voice_dialogue(self.active_session_user, recognized_text)

                if should_render:
                    self.renderer.render(frame, self.current_user_name, top, right, bottom, left)
                else:
                    cv2.putText(frame, "No Face Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                session_status = f"Active Session: {self.active_session_user}" if self.active_session_user else "Session: Closed"
                cv2.putText(frame, session_status, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if not self.renderer.show_and_check_exit(frame):
                    break

                time.sleep(0.5)
        finally:
            cap.release()
            self.renderer.close()
