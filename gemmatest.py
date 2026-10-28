import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "google/gemma-4-E4B-it"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Loading model tensors directly to GPU...")
# Explicitly force the model parameters to skip meta-device allocation artifacts
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,   # Matches Gemma 4's native training precision
    low_cpu_mem_usage=True
)

# Manually send the unified architecture to your CUDA device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Model successfully anchored to: {model.device}")


# Prompt
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short joke about saving RAM."},
]

# Process input
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
outputs = model.generate(**inputs, max_new_tokens=1024)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

# Parse output
processor.parse_response(response)

