import os
import io
import json
import time
import base64
import requests
import pygame
from interfaces import ISpeechSynthesizer

class YandexSpeechKitSynthesizer(ISpeechSynthesizer):
    """
    Класс синтеза речи через современный Yandex SpeechKit API v3.
    Использует потоковую сборку аудио-фрагментов и плеер Pygame.
    """
    def __init__(self, api_key: str, folder_id: str):
        self._secret = api_key
        self._folderId = folder_id

    def speak(self, text: str) -> None:
        """Реализация метода speak для интерфейса ISpeechSynthesizer"""
        if not text or not text.strip():
            return

        print(f"[Yandex TTS v3] Озвучиваю: \"{text}\"")

        url = "https://tts.api.cloud.yandex.net:443/tts/v3/utteranceSynthesis"
        headers = {
            "Authorization": f"Api-Key {self._secret}",
            "x-folder-id": f"{self._folderId}",
            "Content-Type": "application/json"
        }

        payload = {
            "text": text,
            "outputAudioSpec": {
                "containerSpec": {
                    "wav": {}
                }
            },
            "hints": [
                {"voice": "alena"},
                {"speed": 1.05},
                {"role": "good"}
            ]
        }

        try:
            response = requests.post(url, headers=headers, json=payload, stream=True)
            if response.status_code != 200:
                print(f"[Yandex TTS v3] Ошибка API: {response.text}")
                return

            audio_buffer = io.BytesIO()

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    raw_base64 = chunk.get('result', {}).get('audioChunk', {}).get('data')
                    
                    if raw_base64:
                        audio_buffer.write(base64.b64decode(raw_base64))

            audio_buffer.seek(0)
            
            pygame.mixer.init()
            pygame.mixer.music.load(audio_buffer)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
            print("[Yandex TTS v3] Воспроизведение успешно завершено.")

        except Exception as e:
            print(f"[Yandex TTS v3] Критическая ошибка синтеза: {e}")
