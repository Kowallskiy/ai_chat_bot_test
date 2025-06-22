import torch

from text2speech import invoke_tts_model, load_tts
from speech2text import client_speech_to_text, load_speech2text
from classification import load_classification_model, perform_input_classification
from language_model import load_llm_model, generate_llm_text

device = "cuda" if torch.cuda.is_available() else "cpu"
# Загружаем speech2text модель
speech2text_model, rec = load_speech2text()

# Загружаем TTS модель и токенизатор
tts_model, tts_tokenizer = load_tts()

# Загружаем модель классификации
cls_model, cls_tokenizer = load_classification_model(device)

# Разогреем модель для ускорения работы на инференсе
for _ in range(3):
    perform_input_classification("просто тестовый промпт", cls_model, cls_tokenizer, device)

# Загружаем модель LLM
llm_model, llm_tokenizer = load_llm_model()

# Разогрев
for _ in range(2):
    response = generate_llm_text("просто тест", llm_model, llm_tokenizer)
    print(response)

# Контекст для LLM
contexts = [
    "Ты дружелюбный ассистент. Представься и задай один вопрос, чтобы начать знакомство.",
    "Спроси клиента что-нибудь про его повседневную жизнь.",
    "Спроси, как проходит день дома и есть ли питомцы.",
    "Уточни, как человек проводит свободное время.",
    "Спроси, бывает ли сложно с финансами и хватает ли на нужные вещи.",
    "Попроси охарактеризовать текущую жизненную ситуацию.",
    "Спроси, о чём человек мечтает или к чему стремится."
]

dog_trigger = "У нас есть отличное предложение, ошейник для собаки. Вам интересно узнать больше?"
money_trigger = "Для людей с финансовыми трудностями у нас есть отличное решение - MLM. Настоятельно рекомендую Вам поучаствовать."

def start_conversation_loop(prompt: str) -> int:
    """Озвучить вопрос, принять ответ клиента, классифицировать ответ клиента."""
    invoke_tts_model(prompt=prompt, model=tts_model, tokenizer=tts_tokenizer)
    client_response = client_speech_to_text(rec=rec)
    label = perform_input_classification(client_response, cls_model, cls_tokenizer, device)
    return label

def main():
    """Модель начинает цикл общения со сгенерированного LLM вопроса для клиента.
    Вопрос для клиента -> 
    -> озвучить вопрос (TTS) ->
    -> ответ клиента (speech2text) ->
    -> классификация ответа (BERT) ->
    -> возвращаем label ответа клиента, в котором содержится информация об упоминании собаки или 
    отсутствие денег -> 
    -> языковая модель генерирует новый вопрос в зависимости от label ->
    -> цикл начинается сначала
    """
    for llm_context in contexts:
        question_for_client = generate_llm_text(llm_context, llm_model, llm_tokenizer)
        label = start_conversation_loop(question_for_client)
        # label == 1 означает, что была упомянута собака
        if label == 1:
            start_conversation_loop(prompt=dog_trigger)
        # label == 2 означает, что было упомянуто отсутствие денег
        elif label == 2:
            start_conversation_loop(prompt=money_trigger)


if __name__ == "__main__":
    main()