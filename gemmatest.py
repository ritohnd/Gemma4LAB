import os
# Keeps memory segments clean
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import pipeline, BitsAndBytesConfig, AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "google/gemma-4-E4B-it"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# 4-bit compression setup
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("Loading model explicitly to bypass pipeline allocation spikes...")
# Instantiating the model first guarantees that quantization maps correctly to your 8GB VRAM
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",  # Let accelerate handle layer placement
    low_cpu_mem_usage=True
)

print("Initializing Gemma 4 pipeline with preloaded model...")
pipe = pipeline(
    task="any-to-any",
    model=model,
    processor=processor
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short joke about saving RAM."},
]

print("Generating response...")
outputs = pipe(
    messages, 
    max_new_tokens=64,
    generate_kwargs={
        "enable_thinking": False,
    }
)

print("\n--- Model Output ---")
print(outputs[0]["generated_text"])
