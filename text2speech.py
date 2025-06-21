from transformers import VitsModel, AutoTokenizer
import torch
import sounddevice as sd

def load_tts(path: str = "models/tts"):
    try:
        tts_model = VitsModel.from_pretrained(path)
        tts_tokenizer = AutoTokenizer.from_pretrained(path)
        return tts_model, tts_tokenizer
    except Exception as e:
        print(f"Error = {e}")
        return

def invoke_tts_model(prompt: str, model: VitsModel, tokenizer: AutoTokenizer):
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        waveform = model(**inputs).waveform

    wave_np = waveform[0].cpu().numpy()

    sr = model.config.sampling_rate
    sd.play(wave_np, samplerate=sr)
    sd.wait()
