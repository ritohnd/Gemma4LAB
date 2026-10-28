import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig

MODEL_ID = "google/gemma-4-E4B-it"

# 1. Load processor
print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# 2. Configure 4-bit quantization to fit in an 8GB GPU
print("Setting up 4-bit compression config...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# 3. Load the model with quantization mappings
print("Loading model parameters (Compressed 4-bit mode)...")
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
    device_map="auto" # bitsandbytes natively requires device_map="auto" to map quantized blocks
)
print("Model compressed and loaded safely within VRAM limits.")

# Prompt
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short joke about saving RAM."},
]

# 4. Map the inputs using the current model device
print("Processing template tokens...")
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    enable_thinking=False
).to(model.device)

input_len = inputs["input_ids"].shape[-1]

# Generate output
print("Generating response...")
outputs = model.generate(**inputs, max_new_tokens=1024)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

# Parse output
print("\n--- Model Output ---")
parsed_response = processor.parse_response(response)
print(parsed_response)
