import os
# Keeps memory segments clean
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig

MODEL_ID = "google/gemma-4-E4B-it"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# 4-bit configuration with explicit CPU offload enabled
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True  # Allows weights that don't fit in VRAM to sit in system RAM
)

print("Loading model with custom device routing...")
# We build a custom device map where everything goes to GPU 0 by default, 
# but fallback layers can safely spill over to the CPU.
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
    low_cpu_mem_usage=True
)

# Setup prompt messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short joke about saving RAM."},
]

print("Processing input template...")
# Note: For Gemma 4 Multimodal VLMs, the pipeline wrapper can still cause internal spikes.
# Using the native generate workflow ensures strict adherence to our memory config.
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    enable_thinking=False
).to("cuda:0")  # Move tokenized inputs directly onto the primary GPU

input_len = inputs["input_ids"].shape[-1]

print("Generating response...")
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=64
    )

# Extract and decode the newly generated tokens
response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

print("\n--- Model Output ---")
print(response)
