import os
# Force PyTorch to handle memory segments cleanly and prevent the 6.43 GB pre-allocation spike
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import pipeline, BitsAndBytesConfig, AutoProcessor

MODEL_ID = "google/gemma-4-E4B-it"

print("Configuring memory-capped processor constraints...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# 4-bit compression setup
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("Initializing Gemma 4 pipeline directly onto target VRAM...")
pipe = pipeline(
    task="any-to-any",
    model=MODEL_ID,
    processor=processor,
    model_kwargs={
        "quantization_config": quantization_config, 
        "low_cpu_mem_usage": True,
    },
    device_map="cuda:0" # Explicitly maps to GPU 0 to bypass fallback tracing loops
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
        "max_length": 128
    }
)

print("\n--- Model Output ---")
print(outputs["generated_text"])
