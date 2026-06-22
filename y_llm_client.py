import json
import urllib.request
import urllib.error
import ssl
import socket
from interfaces import ILLMClient

class YandexGPTLLMClient(ILLMClient):
    """
    Клиент для работы с текстовой нейросетью YandexGPT v3 API.
    Принимает структурированный контекст диалога и возвращает ответ.
    """
    def __init__(self, api_key: str, folder_id: str, model_type: str = "yandexgpt/latest"):
        self._secret = api_key
        self._folderId = folder_id
        
        self.model_uri = f"gpt://{self._folderId}/{model_type}"
        self.url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    def generate_response(self, prompt_messages: list[dict]) -> str:

        json_payload = {
            "modelUri": f"gpt://{self._folderId}/yandexgpt/latest",
            "completionOptions": {
                "stream": False,
                "temperature": 0.5,
                "maxTokens": 2000
            },
            "messages": []
        }

        for msg in prompt_messages:
            json_payload["messages"].append({
                "role": msg["role"],
                "text": msg["text"]
            })

        request_body = json.dumps(json_payload).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self._secret}"
        }

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        old_getaddrinfo = socket.getaddrinfo
        def ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            return old_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
        
        socket.getaddrinfo = ipv4_getaddrinfo

        try:
            req = urllib.request.Request(self.url, data=request_body, headers=headers, method="POST")
            
            with urllib.request.urlopen(req, context=ctx, timeout=30) as response:
                response_data = response.read().decode('utf-8')
                http_code = response.getcode()

        except urllib.error.HTTPError as e:
            http_code = e.code
            error_data = e.read().decode('utf-8')
            print(f"[YandexGPT AI Error] Yandex Cloud returned HTTP status: {http_code}")
            print(f"[Response Body]: {error_data}")
            return ""
        except Exception as e:
            print(f"[YandexGPT AI Error] Network error или таймаут: {e}")
            return ""
        finally:
            socket.getaddrinfo = old_getaddrinfo

        try:
            parsed_res = json.loads(response_data)

            if ("result" in parsed_res and 
                "alternatives" in parsed_res["result"] and 
                len(parsed_res["result"]["alternatives"]) > 0):

                return parsed_res["result"]["alternatives"][0]["message"]["text"]
            else:
                print("[YandexGPT AI Error] Unexpected JSON format from Yandex API.")
                return ""
                
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"[YandexGPT AI Error] JSON parsing failed: {e}")
            return ""