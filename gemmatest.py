import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "google/gemma-4-E4B-it"

# 1. Load the processor natively
print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# 2. Load model directly onto your hardware without meta-device partitioning
print("Loading model layers...")
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,  # Force native BF16 precision 
    low_cpu_mem_usage=False      # Prevents leaving lazy initialization footprints on 'meta'
)

# 3. Explicitly migrate the model to your GPU device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Model successfully loaded and anchored to: {model.device}")

# Prompt
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short joke about saving RAM."},
]

# 4. Map the inputs directly to our explicit target device variable
print("Processing template tokens...")
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    enable_thinking=False
).to(device) # <--- FIX: Using our direct string variable 'device' here instead of 'model.device'

input_len = inputs["input_ids"].shape[-1]

# Generate output
print("Generating response...")
outputs = model.generate(**inputs, max_new_tokens=1024)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

# Parse output
print("\n--- Model Output ---")
parsed_response = processor.parse_response(response)
print(parsed_response)
