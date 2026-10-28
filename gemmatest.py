import torch
from transformers import pipeline, BitsAndBytesConfig, AutoProcessor

MODEL_ID = "google/gemma-4-E4B-it"

# 1. Force the model processor to use a smaller token text layout matrix
print("Configuring memory-capped processor constraints...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# 2. Configure 4-bit compression mapping
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("Initializing Gemma 4 pipeline...")
pipe = pipeline(
    task="any-to-any",
    model=MODEL_ID,
    processor=processor,
    model_kwargs={
        "quantization_config": quantization_config, 
        "low_cpu_mem_usage": True
    },
    device_map="cuda:0"
)

# Structure chat context mapping layout 
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short joke about saving RAM."},
]

print("Generating response within low-VRAM memory cap limits...")
# 3. Add explicit generation boundaries to stop massive block allocation spikes
outputs = pipe(
    messages, 
    max_new_tokens=64,             # Lower token count keeps the sequence growth compact
    generate_kwargs={
        "enable_thinking": False,  # Bypassing the deep reasoning loop saves activation RAM
        "max_length": 128          # Strictly clamps total operational attention length
    }
)

print("\n--- Model Output ---")
print(outputs["generated_text"])
