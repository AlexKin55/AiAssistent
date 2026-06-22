import time
import grpc
import pyaudio

# СТРОГИЙ ИМПОРТ: в v3 Яндекса сервисы лежат именно в stt_service_pb2_grpc
from yandex.cloud.ai.stt.v3 import stt_pb2
from yandex.cloud.ai.stt.v3 import stt_service_pb2_grpc

from interfaces import IAudioProcessor

class YandexSpeechKitProcessor(IAudioProcessor):

    def __init__(self, api_key: str, folder_id: str):
        self._secret = api_key
        self._folderId = folder_id
        
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 8000            # 8 кГц из вашего примера
        self.CHUNK = 1024
        self.RECORD_SECONDS = 5     # 5 секунд записи из вашего примера
        
        self._audio = pyaudio.PyAudio()

    def _generate_requests(self, stream):
        recognize_options = stt_pb2.StreamingOptions(
            recognition_model=stt_pb2.RecognitionModelOptions(
                audio_format=stt_pb2.AudioFormatOptions(
                    raw_audio=stt_pb2.RawAudio(
                        audio_encoding=stt_pb2.RawAudio.LINEAR16_PCM,
                        sample_rate_hertz=self.RATE,
                        audio_channel_count=self.CHANNELS
                    )
                ),
                text_normalization=stt_pb2.TextNormalizationOptions(
                    text_normalization=stt_pb2.TextNormalizationOptions.TEXT_NORMALIZATION_ENABLED,
                    profanity_filter=True,
                    literature_text=False
                ),
                language_restriction=stt_pb2.LanguageRestrictionOptions(
                    restriction_type=stt_pb2.LanguageRestrictionOptions.WHITELIST,
                    language_code=['ru-RU']
                ),
                audio_processing_type=stt_pb2.RecognitionModelOptions.REAL_TIME
            )
        )

        yield stt_pb2.StreamingRequest(session_options=recognize_options)

        print("[Микрофон] Запись пошла...")
        
        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            try:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                yield stt_pb2.StreamingRequest(chunk=stt_pb2.AudioChunk(data=data))
            except Exception as e:
                print(f"[Микрофон] Ошибка чтения чанка: {e}")
                break
                
        print("[Микрофон] Запись завершена.")

    def listen_and_transcribe(self) -> str:
        """Открывает gRPC сессию, стримит звук и возвращает распознанный текст"""
        try:
            stream = self._audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
        except Exception as e:
            print(f"open audio failed: {e}")
            return ""

        host = "stt.api.cloud.yandex.net:443"
        cred = grpc.ssl_channel_credentials()
        channel = grpc.secure_channel(host, cred)
        stub = stt_service_pb2_grpc.RecognizerStub(channel)

        metadata = (
            ('authorization', f'Api-Key {self._secret}'),
            ('x-folder-id', self._folderId),
        )

        final_text = ""

        try:
            responses = stub.RecognizeStreaming(self._generate_requests(stream), metadata=metadata)
            for response in responses:
                if response.HasField("final_refinement"):
                    alt = response.final_refinement.normalized_text.alternatives
                    if alt:
                        final_text = alt[0].text

        except Exception as e:
            print(f"[Yandex STT v3] Ошибка gRPC стриминга: {e}")
        finally:
            stream.stop_stream()
            stream.close()

        return final_text.strip()
