import torch
from transformers import pipeline, BitsAndBytesConfig, AutoProcessor

MODEL_ID = "google/gemma-4-E4B-it"

# 1. Force lower vision/text allocation limits on the processor configuration
print("Configuring aggressive memory caps...")
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    max_image_tokens=140,       # Sets a lower visual budget to drastically cut activation VRAM
    max_sequence_length=512     # Constrains the text token buffer size
)

# 2. Configure 4-bit compression
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("Initializing capped Gemma 4 pipeline...")
pipe = pipeline(
    task="any-to-any",
    model=MODEL_ID,
    processor=processor, # Binds our custom low-VRAM memory cap configuration
    model_kwargs={
        "quantization_config": quantization_config, 
        "low_cpu_mem_usage": True
    },
    device_map="cuda:0"
)

# Structure chat prompt
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short joke about saving RAM."},
]

print("Generating response within safe VRAM boundaries...")
# Set strict limits on token generation to prevent runtime allocation expansion
outputs = pipe(
    messages, 
    max_new_tokens=128, # Lower generation length cuts runtime context memory growth
    generate_kwargs={"enable_thinking": False}
)

print("\n--- Model Output ---")
print(outputs["generated_text"])
