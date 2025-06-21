import queue, sys
import sounddevice as sd
from sounddevice import CallbackFlags
from vosk import KaldiRecognizer, Model
import json

def load_speech2text():
    """загружает speech2text модель и Recognizer"""
    speech2text_model = Model(model_path="models/vosk-model-small-ru-0.22")
    rec = KaldiRecognizer(speech2text_model, 16000)
    return speech2text_model, rec

def client_speech_to_text(rec: KaldiRecognizer) -> str:
    """Слушает ответ клиента в режиме реального времени и переводит результат в текст.

    Возвращает текстовый ответ клиента.
    """
    q = queue.Queue()
    def callback(indata, frames: int, time, status: CallbackFlags):
        """добавляет куски текста в очередь"""
        if status:
            print(status, file=sys.stderr)
        q.put(bytes(indata))
    try:
        # 8000 / 16000 = 0.5s каждые 0.5 с мы вызываем callback
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16",
                               channels=1, callback=callback):
            while True:
                data = q.get()
                if rec.AcceptWaveform(data):
                    final_text = json.loads(rec.Result())
                    return final_text["text"] # {"text": "final text"}
                else:
                    print(rec.PartialResult(), end="\r") # {"partial": "final te##"}
    except Exception as e:
        print(f"Error: {e}")