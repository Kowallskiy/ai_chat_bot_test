import torch

from text2speech import invoke_tts_model, load_tts
from speech2text import client_speech_to_text, load_speech2text
from classification import load_classification_model, perform_input_classification

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

# Это может быть языковая модель, логика та же. Использую список для показательности
initial_prompts = [
    "Здравствуйте! Расскажите немного о себе.",
    "Чем вы обычно занимаетесь в повседневной жизни?",
    "Как проходит ваш обычный день дома? Есть ли у вас домашние животные?",
    "Как вы проводите свободное время?",
    "Бывают ли у вас финансовые трудности? Например, когда не хватает на что-то нужное?",
    "Как бы вы охарактеризовали свою жизнь сейчас?",
    "О чём вы мечтаете или к чему стремитесь?"
]
dog_trigger = "У нас есть отличное предложение, ошейник для собаки. Вам интересно узнать больше?"
money_trigger = "Для людей с финансовыми трудностями у нас есть отличное решение - MLM. Настоятельно рекомендую Ввам поучаствовать."

def start_conversation_loop(prompt: str) -> int:
    """Модель начинает цикл общения со сгенерированного промпта.
    Промпт -> озвучить промпт (TTS) -> ответ клиента (speech2text) -> классификация ответа (BERT) ->
    -> возвращаем label ответа клиента, в котором содержится информация об упоминании собаки или 
    отсутствие денег -> языковая модель генерирует новый промпт в зависимости от label -> цикл начинается сначала
    """
    invoke_tts_model(prompt=prompt, model=tts_model, tokenizer=tts_tokenizer)
    client_response = client_speech_to_text(rec=rec)
    label = perform_input_classification(client_response, cls_model, cls_tokenizer, device)
    return label

def main():
    for prompt in initial_prompts:
        label = start_conversation_loop(prompt)
        # label == 1 означает, что была упомянута собака
        if label == 1:
            start_conversation_loop(prompt=dog_trigger)
        # label == 2 означает, что было упомянуто отсутствие денег
        elif label == 2:
            start_conversation_loop(prompt=money_trigger)


if __name__ == "__main__":
    main()