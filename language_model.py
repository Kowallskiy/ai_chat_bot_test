import os
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from huggingface_hub import login

HF_API = os.getenv("HF_API")
login(token=HF_API)

def load_llm_model(model_name: str = "meta-llama/Llama-3.2-1B"):

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map= "auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer

def generate_llm_text(context: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    encoded = tokenizer(context, return_tensors="pt")
    output = model.generate(encoded["input_ids"], max_length = 50)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    clean_response = response[len(context) + 1:]
    return clean_response
