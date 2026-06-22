from interfaces import IConversation, IUserManager

class RetailAssistantConversation(IConversation):
    def __init__(self, user_manager: IUserManager, system_instruction: str = None):
        self.user_manager = user_manager
        
        # Инструкция по умолчанию, если не передана кастомная
        self.system_instruction = system_instruction or (
            "Ты — вежливый и опытный ассистент-продавец в магазине обуви и одежды. "
            "Твоя задача — консультировать клиента, помогать с выбором и мягко предлагать сопутствующие товары. "
            "Отвечай лаконично, дружелюбно и по делу."
        )

    def build_prompt_messages(self, user_id: str, current_user_message: str) -> list[dict]:
        purchases, chat_history = self.user_manager.get_llm_context(user_id)
        purchases_str = ", ".join(purchases) if purchases else "Клиент еще ничего у нас не покупал."
        
        dynamic_system_prompt = (
            f"{self.system_instruction}\n\n"
            f"КОНТЕКСТ О КЛИЕНТЕ ({user_id}):\n"
            f"- Ранее купленные товары: [{purchases_str}].\n"
            "ПРАВИЛО: Если клиент интересуется вещью, которую он уже покупал, "
            "обязательно начни ответ с вопроса, понравился ли ему этот товар в прошлый раз."
        )

        prompt_messages = [
            {"role": "system", "text": dynamic_system_prompt}
        ]

        for msg in chat_history:
            prompt_messages.append({
                "role": msg["role"], 
                "text": msg["text"]
            })

        prompt_messages.append({
            "role": "user", 
            "text": current_user_message
        })

        return prompt_messages


""" to use
# ПРИМЕР ТОГО, КАК КЛАССЫ РАБОТАЮТ ВМЕСТЕ

def process_user_speech(user_id: str, user_text: str):
    # 1. Через IConversation создаем идеальный промпт со всей подложкой данных
    messages_for_llm = conversation_manager.build_prompt_messages(user_id, user_text)
    
    # 2. Передаем этот массив напрямую в нейросеть (Ollama, YandexGPT или OpenAI)
    # Структура messages_for_llm идеально ложится в стандартные API
    ai_response = my_llm_client.generate(messages_for_llm)
    
    print(f"Робот-Продавец: {ai_response}")
    
    # 3. Фиксируем этот раунд в скользящее окно SQLite через менеджер пользователей
    user_manager.add_message_to_history(user_id, "user", user_text)
    user_manager.add_message_to_history(user_id, "assistant", ai_response)
"""